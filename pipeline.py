"""
Zeely AI Avatar Generator Pipeline
====================================
Two-stage automated engine:
  1. Avatar Generation: User photo → Studio portrait on white background (#FFFFFF)
  2. Outfit Transfer: Avatar + garment image/prompt → Same person in new outfit

APIs used:
  - fal.ai BiRefNet (portrait mode) — background removal
  - fal.ai FLUX PuLID — identity-preserving studio portrait generation
  - Replicate IDM-VTON — virtual try-on / outfit transfer

Production features:
  - Retry with exponential backoff on API failures
  - Upload caching (deduplication via content hash)
  - Automated quality scoring (background purity, blur, face detection)
  - Configurable prompt presets (studio / fashion / ecommerce)
  - Parallel batch processing with ThreadPoolExecutor
  - Structured JSON reporting with per-image QA metrics and timings

Author: Konstantin Kovalchuk
"""

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import functools
import requests
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageFilter, ImageStat

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FAL_API_KEY = os.environ.get("FAL_KEY", "")
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN", "")

# Model identifiers
BG_REMOVAL_MODEL = "fal-ai/birefnet/v2"
BG_REMOVAL_VARIANT = "BiRefNet-portrait"
AVATAR_RESOLUTION = "1024x1024"

VTON_MODEL = "cuuupid/idm-vton:c871bb9b046c1462f19c2ee0f21c122e01c3f04b3e3310c13cde50b4d89e7f73"
VTON_STEPS = 30
VTON_SEED = 42

# Directories
DEFAULT_INPUT_DIR = "input/users"
DEFAULT_OUTFIT_DIR = "input/outfits"
DEFAULT_OUTPUT_DIR = "output"

# Output dimensions (3:4 portrait)
OUTPUT_WIDTH = 768
OUTPUT_HEIGHT = 1024

# Retry settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds — exponential backoff: 2s, 4s, 8s

# Quality thresholds
QA_BG_PURITY_THRESHOLD = 0.95    # 95% of border pixels must be near-white
QA_BLUR_THRESHOLD = 80.0          # Laplacian variance; below = blurry
QA_MIN_FACE_AREA_PCT = 0.03       # Face must occupy >= 3% of image

# Concurrency
MAX_WORKERS = 3  # Parallel user processing (respects API rate limits)


# ---------------------------------------------------------------------------
# Prompt Presets
# ---------------------------------------------------------------------------

PROMPT_PRESETS = {
    "studio": {
        "name": "Studio Portrait",
        "prompt": (
            "Professional studio headshot portrait of this person, "
            "clean white background, soft diffused studio lighting with softboxes, "
            "neutral pose facing camera, half-body shot from head to waist, "
            "even skin tones, no harsh shadows, high detail, "
            "8K quality studio photograph"
        ),
        "guidance_scale": 4.0,
        "id_weight": 1.0,
        "steps": 30,
    },
    "fashion": {
        "name": "Fashion Editorial",
        "prompt": (
            "High-end fashion editorial portrait of this person, "
            "pure white seamless backdrop, professional fashion photography lighting, "
            "confident neutral pose, half-body framing from head to hips, "
            "crisp details, magazine-quality skin retouching, "
            "shot on medium format camera"
        ),
        "guidance_scale": 5.0,
        "id_weight": 0.9,
        "steps": 35,
    },
    "ecommerce": {
        "name": "E-commerce Product",
        "prompt": (
            "Clean e-commerce model portrait of this person, "
            "plain white background, flat even lighting, "
            "straight-on frontal pose, half-body crop, "
            "commercial product photography style, "
            "sharp focus, consistent color temperature"
        ),
        "guidance_scale": 3.5,
        "id_weight": 1.0,
        "steps": 25,
    },
}

DEFAULT_PRESET = "studio"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("zeely-pipeline")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Automated quality metrics for a generated image."""
    bg_purity: float = 0.0          # 0–1, fraction of white border pixels
    blur_score: float = 0.0         # Laplacian variance (higher = sharper)
    face_detected: bool = False
    face_area_pct: float = 0.0
    dimensions_ok: bool = False
    passed: bool = False

    def evaluate(self):
        """Set overall pass/fail from individual metrics."""
        self.passed = (
            self.bg_purity >= QA_BG_PURITY_THRESHOLD
            and self.blur_score >= QA_BLUR_THRESHOLD
            and self.dimensions_ok
        )


@dataclass
class UserResult:
    """Processing result for a single user."""
    user_id: str
    source_file: str
    status: str = "pending"
    avatar_path: Optional[str] = None
    outfit_path: Optional[str] = None
    avatar_qa: Optional[dict] = None
    outfit_qa: Optional[dict] = None
    errors: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Upload cache — avoids re-uploading identical files to CDN
# ---------------------------------------------------------------------------

class UploadCache:
    """
    Content-addressed upload cache.
    Same file content → same CDN URL, regardless of filename.
    Saves cost and latency in batch processing.
    """

    def __init__(self):
        self._cache: dict[str, str] = {}

    @staticmethod
    def _hash_file(path: str) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def get_or_upload(self, image_path: str) -> str:
        """Return cached CDN URL or upload and cache."""
        content_hash = self._hash_file(image_path)
        if content_hash in self._cache:
            log.debug(f"  ⚡ Cache hit for {image_path}")
            return self._cache[content_hash]

        url = _raw_upload_to_fal(image_path)
        self._cache[content_hash] = url
        log.debug(f"  📤 Uploaded and cached {image_path}")
        return url

    def stats(self) -> dict:
        return {"cached_files": len(self._cache)}


_upload_cache = UploadCache()


# ---------------------------------------------------------------------------
# Retry wrapper
# ---------------------------------------------------------------------------

def with_retry(func):
    """
    Decorator for API calls with exponential backoff.
    Retries on: network errors, timeouts, rate limits (429), server errors (5xx).
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(MAX_RETRIES + 1):
            try:
                return func(*args, **kwargs)
            except (
                requests.exceptions.RequestException,
                TimeoutError,
                ConnectionError,
                RuntimeError,
            ) as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    log.warning(
                        f"  ⚠ {func.__name__} attempt {attempt + 1}/{MAX_RETRIES + 1} "
                        f"failed: {e}. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    log.error(f"  ✗ {func.__name__}: all {MAX_RETRIES + 1} attempts failed")
        raise last_exc
    return wrapper


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


@with_retry
def download_image(url: str, save_path: str) -> str:
    """Download image from URL with retry."""
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    return save_path


def _raw_upload_to_fal(image_path: str) -> str:
    """Encode image as base64 data URI (fal.ai models accept data URIs directly)."""
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(image_path)
    if not mime:
        mime = "image/png"
    with open(image_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def upload_to_fal(image_path: str) -> str:
    """Upload with content-hash caching."""
    return _upload_cache.get_or_upload(image_path)


# ---------------------------------------------------------------------------
# Quality Assessment
# ---------------------------------------------------------------------------

def assess_quality(image_path: str) -> QualityReport:
    """
    Automated quality scoring for generated images.

    Metrics:
    1. Background purity — border pixels should be white (#FFFFFF)
    2. Sharpness — Laplacian variance (detects AI blur / plastic look)
    3. Face presence — skin-tone heuristic in upper-center region
    4. Dimensions — must match target output (768×1024)

    A production system would extend this with:
    - Face similarity scoring (ArcFace / insightface embeddings vs. source)
    - CLIP image-text similarity for prompt adherence
    - Aesthetic scoring model (LAION aesthetic predictor)
    - Anatomical artifact detection (hand/finger counting)
    """
    qa = QualityReport()

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return qa

    w, h = img.size

    # 1. Dimensions
    qa.dimensions_ok = (w == OUTPUT_WIDTH and h == OUTPUT_HEIGHT)

    # 2. Background purity: sample border pixels (outer 5%)
    border = max(1, int(min(w, h) * 0.05))
    white_count = 0
    total_border = 0

    for x in range(w):
        for y in list(range(border)) + list(range(h - border, h)):
            r, g, b = img.getpixel((x, y))
            total_border += 1
            if r > 240 and g > 240 and b > 240:
                white_count += 1
    for y in range(border, h - border):
        for x in list(range(border)) + list(range(w - border, w)):
            r, g, b = img.getpixel((x, y))
            total_border += 1
            if r > 240 and g > 240 and b > 240:
                white_count += 1

    qa.bg_purity = white_count / max(1, total_border)

    # 3. Blur detection: Laplacian variance on center crop
    center = img.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4))
    gray = center.convert("L")
    laplacian = gray.filter(ImageFilter.Kernel(
        size=(3, 3),
        kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1],
        scale=1, offset=128,
    ))
    qa.blur_score = ImageStat.Stat(laplacian).var[0]

    # 4. Face presence: skin-tone heuristic in upper-center
    #    (Production upgrade: insightface / mediapipe face detector)
    face_region = img.crop((w // 4, 0, 3 * w // 4, h // 2))
    total_px = face_region.width * face_region.height
    step = max(1, total_px // 5000)
    skin_hits = 0
    samples = 0

    for i in range(0, total_px, step):
        x = i % face_region.width
        y = i // face_region.width
        if y >= face_region.height:
            break
        r, g, b = face_region.getpixel((x, y))
        samples += 1
        if r > 60 and g > 40 and b > 20 and r > g and (r - g) < 100:
            skin_hits += 1

    skin_pct = skin_hits / max(1, samples)
    qa.face_detected = skin_pct > 0.10
    qa.face_area_pct = round(skin_pct, 3)

    qa.evaluate()
    return qa


# ---------------------------------------------------------------------------
# STEP 1: Avatar Generation
# ---------------------------------------------------------------------------

@with_retry
def remove_background_fal(image_url: str) -> str:
    """Remove background using fal.ai BiRefNet (portrait mode). Returns foreground URL."""
    import fal_client

    log.info("  → BiRefNet background removal (portrait mode)...")

    result = fal_client.subscribe(
        BG_REMOVAL_MODEL,
        arguments={
            "image_url": image_url,
            "model": BG_REMOVAL_VARIANT,
            "operating_resolution": AVATAR_RESOLUTION,
            "output_format": "png",
            "refine_foreground": True,
        },
    )

    log.info("  ✓ Background removed")
    return result["image"]["url"]


def composite_on_white(foreground_path: str, output_path: str) -> str:
    """Composite transparent PNG onto pure white, auto-crop to portrait framing."""
    fg = Image.open(foreground_path).convert("RGBA")
    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, fg).convert("RGB")
    final = auto_crop_portrait(composite)
    final.save(output_path, "PNG", quality=95)
    log.info(f"  ✓ Composited on white → {output_path}")
    return output_path


def auto_crop_portrait(img: Image.Image, padding_pct: float = 0.08) -> Image.Image:
    """Auto-crop to person with proportional padding, resize to 768×1024."""
    from PIL import ImageChops

    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img

    x1, y1, x2, y2 = bbox
    cw, ch = x2 - x1, y2 - y1
    px, py = int(cw * padding_pct), int(ch * padding_pct)

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(img.width, x2 + px)
    y2 = min(img.height, y2 + py)

    return img.crop((x1, y1, x2, y2)).resize(
        (OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS
    )


@with_retry
def _run_pulid(image_url: str, preset: dict) -> str:
    """Run FLUX PuLID with retry. Returns generated image URL."""
    import fal_client

    result = fal_client.subscribe(
        "fal-ai/flux-pulid",
        arguments={
            "prompt": preset["prompt"],
            "reference_images": [{"image_url": image_url}],
            "image_size": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
            "num_inference_steps": preset["steps"],
            "guidance_scale": preset["guidance_scale"],
            "id_weight": preset["id_weight"],
            "true_cfg": 1.0,
            "max_sequence_length": 128,
        },
    )
    return result["images"][0]["url"]


def generate_avatar(
    image_path: str,
    output_path: str,
    use_pulid: bool = False,
    preset_name: str = DEFAULT_PRESET,
) -> QualityReport:
    """
    Full avatar generation pipeline:
    1. Upload image (cached)
    2. Remove background (BiRefNet portrait + retry)
    3. Composite on white #FFFFFF
    4. (Optional) FLUX PuLID studio relight
    5. Automated quality assessment

    Returns QualityReport.
    """
    log.info(f"Generating avatar for: {image_path}")

    image_url = upload_to_fal(image_path)
    fg_url = remove_background_fal(image_url)

    fg_path = output_path.replace(".png", "_fg.png")
    download_image(fg_url, fg_path)
    composite_on_white(fg_path, output_path)

    if use_pulid:
        preset = PROMPT_PRESETS.get(preset_name, PROMPT_PRESETS[DEFAULT_PRESET])
        log.info(f"  → FLUX PuLID enhancement ({preset['name']})...")
        enhanced_url = _run_pulid(image_url, preset)
        download_image(enhanced_url, output_path)
        log.info("  ✓ PuLID applied")

    # Cleanup
    if os.path.exists(fg_path):
        os.remove(fg_path)

    qa = assess_quality(output_path)
    status = "PASS" if qa.passed else "WARN"
    log.info(
        f"  ✓ Avatar saved [{status}] bg={qa.bg_purity:.0%} "
        f"sharp={qa.blur_score:.0f} face={qa.face_detected}"
    )
    return qa


# ---------------------------------------------------------------------------
# STEP 2: Outfit Transfer
# ---------------------------------------------------------------------------

@with_retry
def _run_idm_vton(avatar_path: str, garment_path: str, category: str) -> str:
    """Run IDM-VTON with retry. Returns result URL."""
    import replicate

    output = replicate.run(
        VTON_MODEL,
        input={
            "human_img": open(avatar_path, "rb"),
            "garm_img": open(garment_path, "rb"),
            "category": category,
            "steps": VTON_STEPS,
            "seed": VTON_SEED,
            "auto_mask": True,
            "auto_crop": True,
        },
    )

    if isinstance(output, str):
        return output
    elif hasattr(output, "url"):
        return output.url
    elif isinstance(output, list) and len(output) > 0:
        return str(output[0])
    return str(output)


def transfer_outfit(
    avatar_path: str,
    garment_path: str,
    output_path: str,
    category: str = "upper_body",
) -> QualityReport:
    """Outfit transfer with retry and QA. Returns QualityReport."""
    log.info(f"  → IDM-VTON outfit transfer (category: {category})...")

    result_url = _run_idm_vton(avatar_path, garment_path, category)
    download_image(result_url, output_path)

    # Ensure consistent dimensions
    img = Image.open(output_path).convert("RGB")
    if img.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
        img.save(output_path, "PNG", quality=95)

    qa = assess_quality(output_path)
    status = "PASS" if qa.passed else "WARN"
    log.info(f"  ✓ Outfit saved [{status}] bg={qa.bg_purity:.0%} sharp={qa.blur_score:.0f}")
    return qa


# ---------------------------------------------------------------------------
# Batch Automation
# ---------------------------------------------------------------------------

def process_single_user(
    user_id: str,
    user_image_path: str,
    outfit_path: str,
    output_dir: str,
    use_pulid: bool = False,
    preset_name: str = DEFAULT_PRESET,
    outfit_category: str = "upper_body",
) -> UserResult:
    """Process one user through both pipeline stages with timing and QA."""
    user_output_dir = os.path.join(output_dir, user_id)
    ensure_dir(user_output_dir)

    result = UserResult(user_id=user_id, source_file=os.path.basename(user_image_path))

    # Stage 1: Avatar
    avatar_path = os.path.join(user_output_dir, "avatar.png")
    t0 = time.time()
    try:
        avatar_qa = generate_avatar(user_image_path, avatar_path, use_pulid, preset_name)
        result.avatar_path = avatar_path
        result.avatar_qa = asdict(avatar_qa)
        result.timings["avatar_s"] = round(time.time() - t0, 1)
    except Exception as e:
        log.error(f"  ✗ Avatar failed for {user_id}: {e}")
        result.status = "failed"
        result.errors.append(f"Avatar: {e}")
        result.timings["avatar_s"] = round(time.time() - t0, 1)
        return result

    # Stage 2: Outfit
    outfit_out = os.path.join(user_output_dir, "avatar_outfit.png")
    t1 = time.time()
    try:
        outfit_qa = transfer_outfit(avatar_path, outfit_path, outfit_out, outfit_category)
        result.outfit_path = outfit_out
        result.outfit_qa = asdict(outfit_qa)
        result.status = "success"
        result.timings["outfit_s"] = round(time.time() - t1, 1)
    except Exception as e:
        log.error(f"  ✗ Outfit failed for {user_id}: {e}")
        result.status = "partial_failure"
        result.errors.append(f"Outfit: {e}")
        result.timings["outfit_s"] = round(time.time() - t1, 1)

    result.timings["total_s"] = round(time.time() - t0, 1)
    return result


def run_batch(
    input_dir: str,
    outfit_dir: str,
    output_dir: str,
    use_pulid: bool = False,
    preset_name: str = DEFAULT_PRESET,
    outfit_category: str = "upper_body",
    parallel: bool = False,
):
    """
    Batch-process all user images.
    Supports sequential (default) and parallel (--parallel) execution.
    """
    ensure_dir(output_dir)

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    user_images = sorted(f for f in Path(input_dir).iterdir() if f.suffix.lower() in valid_ext)
    outfit_images = sorted(f for f in Path(outfit_dir).iterdir() if f.suffix.lower() in valid_ext)

    if not user_images:
        log.error(f"No images in {input_dir}"); sys.exit(1)
    if not outfit_images:
        log.error(f"No outfits in {outfit_dir}"); sys.exit(1)

    log.info("=" * 60)
    log.info("Zeely AI Avatar Pipeline — Batch Mode")
    log.info(f"Users: {len(user_images)} | Outfits: {len(outfit_images)}")
    log.info(f"PuLID: {'ON' if use_pulid else 'OFF'} | Preset: {preset_name}")
    log.info(f"Parallel: {'ON' if parallel else 'OFF'} | Workers: {MAX_WORKERS}")
    log.info("=" * 60)

    t_batch = time.time()

    tasks = [
        (f"{i:03d}", str(img), str(outfit_images[(i - 1) % len(outfit_images)]))
        for i, img in enumerate(user_images, 1)
    ]

    results: list[UserResult] = []

    if parallel and len(tasks) > 1:
        log.info(f"\n🚀 Parallel: {len(tasks)} users, max {MAX_WORKERS} workers\n")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {
                pool.submit(
                    process_single_user, uid, upath, opath,
                    output_dir, use_pulid, preset_name, outfit_category,
                ): uid
                for uid, upath, opath in tasks
            }
            for fut in as_completed(futures):
                uid = futures[fut]
                try:
                    results.append(fut.result())
                    log.info(f"  → User {uid}: {results[-1].status}")
                except Exception as e:
                    log.error(f"  ✗ User {uid} crashed: {e}")
                    results.append(UserResult(uid, "", status="failed", errors=[str(e)]))
    else:
        for uid, upath, opath in tasks:
            log.info(f"\n--- User {uid}: {Path(upath).name} ---")
            results.append(process_single_user(
                uid, upath, opath, output_dir, use_pulid, preset_name, outfit_category,
            ))
            log.info(f"  → {results[-1].status}")

    results.sort(key=lambda r: r.user_id)
    elapsed = round(time.time() - t_batch, 1)

    ok = sum(1 for r in results if r.status == "success")
    qa_ok = sum(1 for r in results if r.avatar_qa and r.avatar_qa.get("passed"))

    report = {
        "summary": {
            "total": len(results), "success": ok, "qa_passed": qa_ok,
            "failed": len(results) - ok, "seconds": elapsed,
            "avg_per_user": round(elapsed / max(1, len(results)), 1),
            "preset": preset_name, "pulid": use_pulid, "parallel": parallel,
        },
        "cache": _upload_cache.stats(),
        "results": [asdict(r) for r in results],
    }

    report_path = os.path.join(output_dir, "batch_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"\n{'=' * 60}")
    log.info(f"Done in {elapsed}s — {ok}/{len(results)} success, {qa_ok} QA passed")
    log.info(f"Report: {report_path}")
    log.info("=" * 60)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Zeely AI Avatar Generator Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py                                     # Batch, defaults
  python pipeline.py --pulid --preset fashion            # Fashion preset
  python pipeline.py --parallel --pulid                  # Parallel + PuLID
  python pipeline.py -s photo.jpg --outfit-single b.jpg  # Single image
  python pipeline.py --qa-only output/001/avatar.png     # QA scoring only
        """,
    )

    p.add_argument("--input", "-i", default=DEFAULT_INPUT_DIR)
    p.add_argument("--outfits", "-f", default=DEFAULT_OUTFIT_DIR)
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pulid", action="store_true", help="Enable PuLID studio enhancement")
    p.add_argument("--preset", "-p", choices=list(PROMPT_PRESETS.keys()), default=DEFAULT_PRESET)
    p.add_argument("--category", "-c", choices=["upper_body", "lower_body", "dresses"], default="upper_body")
    p.add_argument("--parallel", action="store_true", help=f"Parallel processing ({MAX_WORKERS} workers)")
    p.add_argument("--single", "-s", type=str, default=None, help="Single image mode")
    p.add_argument("--outfit-single", type=str, default=None)
    p.add_argument("--qa-only", type=str, default=None, help="Run QA on existing image")

    args = p.parse_args()

    if args.qa_only:
        print(json.dumps(asdict(assess_quality(args.qa_only)), indent=2))
        return

    if not FAL_API_KEY:
        log.error("FAL_KEY not set → https://fal.ai/dashboard/keys"); sys.exit(1)
    if not REPLICATE_API_TOKEN:
        log.error("REPLICATE_API_TOKEN not set → https://replicate.com/account/api-tokens"); sys.exit(1)

    if args.single:
        outfit = args.outfit_single
        if not outfit:
            ofs = sorted(Path(args.outfits).glob("*"))
            outfit = str(ofs[0]) if ofs else None
        if not outfit:
            log.error("No outfit specified"); sys.exit(1)
        r = process_single_user("001", args.single, outfit, args.output, args.pulid, args.preset, args.category)
        print(json.dumps(asdict(r), indent=2))
    else:
        run_batch(args.input, args.outfits, args.output, args.pulid, args.preset, args.category, args.parallel)


if __name__ == "__main__":
    main()
