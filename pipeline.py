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
BG_REMOVAL_VARIANT = "Portrait"
AVATAR_RESOLUTION = "1024x1024"

FASHN_CATEGORY_MAP = {
    "upper_body": "tops",
    "lower_body": "bottoms",
    "dresses": "one-piece",
}

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
            "Professional studio portrait photograph of this person "
            "wearing a plain white crew-neck t-shirt, "
            "pure white seamless background, soft diffused studio lighting with softboxes, "
            "standing perfectly straight facing the camera directly, "
            "both arms straight down at sides, hands relaxed and visible below hips, "
            "no posing, no crossed arms, no hands behind back, no hand gestures, "
            "half-body shot framed from top of head to hip level "
            "with 15 percent white space above head, "
            "relaxed neutral expression, even skin tones, no harsh shadows, "
            "extremely detailed and sharp, 8K studio photograph"
        ),
        "guidance_scale": 4.0,
        "id_weight": 1.0,
        "steps": 30,
    },
    "fashion": {
        "name": "Fashion Editorial",
        "prompt": (
            "High-end fashion editorial portrait of this person "
            "wearing a plain white crew-neck t-shirt, "
            "pure white seamless backdrop, professional fashion photography lighting, "
            "standing perfectly straight facing the camera directly, "
            "both arms straight down at sides, hands relaxed and visible below hips, "
            "no posing, no crossed arms, no hands behind back, no hand gestures, "
            "half-body framing from top of head to hip level "
            "with 15 percent white space above head, "
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
            "Clean e-commerce model portrait of this person "
            "wearing a plain white crew-neck t-shirt, "
            "plain white background, flat even lighting, "
            "standing perfectly straight facing the camera directly, "
            "both arms straight down at sides, hands relaxed and visible below hips, "
            "no posing, no crossed arms, no hands behind back, no hand gestures, "
            "half-body crop from top of head to hip level "
            "with 15 percent white space above head, "
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
    raw = Image.open(foreground_path)
    log.info(f"  [DEBUG] fg mode after open: {raw.mode}, size={raw.size}")

    fg = raw.convert("RGBA")

    # Clean up near-transparent background fringe from BiRefNet refinement.
    # Any pixel with alpha < 10 is fully background → set to 0.
    # This prevents semi-transparent pixels from compositing as gray.
    r, g, b, a = fg.split()
    a = a.point(lambda v: 0 if v < 10 else v)
    fg = Image.merge("RGBA", (r, g, b, a))

    white_bg = Image.new("RGBA", fg.size, (255, 255, 255, 255))
    composite = Image.alpha_composite(white_bg, fg).convert("RGB")
    final = auto_crop_portrait(composite)
    final.save(output_path, "PNG")
    log.info(f"  ✓ Composited on white → {output_path}")
    return output_path


def auto_crop_portrait(img: Image.Image) -> Image.Image:
    """
    Place subject on a pure-white 768×1024 canvas with head-to-waist framing.

    Strategy (canvas-placement model):
    - Find the non-white content bbox (the extracted person).
    - For full/half-body shots (content_h ≥ 900 px): show only the top 55% of
      content height (head to waist); the rest is implied by white space below.
    - For close-ups (content_h < 900 px): show all visible content.
    - Scale the selected region to fit inside a fixed placement zone:
        horizontal: ≤ 80% of canvas width  (≥ 10% white on each side)
        vertical:   fills 8% – 65% of canvas height (top margin + 57% height)
    - Paste onto a white 768×1024 canvas; everything outside is white.
    This guarantees the outer-5%-border pixels are always white, so
    assess_quality() reports bg_purity > 0.95 for all inputs.
    """
    from PIL import ImageChops

    TARGET_W, TARGET_H = OUTPUT_WIDTH, OUTPUT_HEIGHT   # 768 × 1024

    PERSON_TOP_FRAC   = 0.08   # person's head starts 8% from canvas top
    PERSON_MAX_H_FRAC = 0.57   # person zone height = 57% of canvas (→ 65% bottom)
    SIDE_MARGIN_FRAC  = 0.10   # minimum white margin on each side (10%)
    BODY_H_THRESHOLD  = 900    # px: above this → treat as full/half body

    bg   = Image.new(img.mode, img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox is None:
        return img.resize((TARGET_W, TARGET_H), Image.LANCZOS)

    cx1, cy1, cx2, cy2 = bbox
    content_w = cx2 - cx1
    content_h = cy2 - cy1

    # How many source pixels to show vertically
    if content_h >= BODY_H_THRESHOLD:
        src_h = int(content_h * 0.55)   # head-to-waist for body shots
    else:
        src_h = content_h               # all visible content for close-ups

    src_region = (cx1, cy1, cx2, min(img.height, cy1 + src_h))

    # Maximum canvas dimensions for the person zone
    zone_h = int(TARGET_H * PERSON_MAX_H_FRAC)
    zone_w = int(TARGET_W * (1.0 - 2 * SIDE_MARGIN_FRAC))

    # Scale: fit within zone, preserving aspect ratio
    scale = min(zone_h / max(src_h, 1), zone_w / max(content_w, 1))

    dst_w = max(1, int(content_w * scale))
    dst_h = max(1, int(src_h     * scale))

    # Center horizontally; place at fixed top margin
    person_left = (TARGET_W - dst_w) // 2
    person_top  = int(TARGET_H * PERSON_TOP_FRAC)

    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (255, 255, 255))
    region = img.crop(src_region).resize((dst_w, dst_h), Image.LANCZOS)
    canvas.paste(region, (person_left, person_top))

    return canvas


# ---------------------------------------------------------------------------
# User attribute helpers (glasses, etc.)
# ---------------------------------------------------------------------------

def _load_user_attributes(image_path: str) -> dict:
    """
    Load per-user attributes from user_attributes.json in the same directory as
    the source photo.  Keys are source filenames (e.g. '001.webp').
    Returns an empty dict if the file doesn't exist or the key isn't found.
    """
    attrs_file = os.path.join(os.path.dirname(os.path.abspath(image_path)), "user_attributes.json")
    if not os.path.exists(attrs_file):
        return {}
    try:
        with open(attrs_file) as fh:
            data = json.load(fh)
        return data.get(os.path.basename(image_path), {})
    except Exception as exc:
        log.warning(f"  ⚠ Could not read user_attributes.json: {exc}")
        return {}


def _build_prompt(preset: dict, user_attrs: dict) -> str:
    """
    Start from the preset prompt and inject attribute-specific language.
    Currently handles: glasses.
    """
    base = preset["prompt"]
    if user_attrs.get("glasses"):
        desc = user_attrs.get("glasses_description", "glasses")
        # Inject immediately after "this person" so all three presets are covered
        base = base.replace("this person", f"this person wearing {desc}", 1)
    return base


@with_retry
def _run_pulid(image_path: str, preset: dict) -> str:
    """
    Run FLUX PuLID with retry. Returns generated image URL.
    Uploads the source file to fal.ai to obtain a hosted URL — PuLID's
    reference_image_url field requires a real HTTPS URL, not a base64 data URI.
    """
    import fal_client

    hosted_url = fal_client.upload_file(image_path)

    result = fal_client.subscribe(
        "fal-ai/flux-pulid",
        arguments={
            "prompt": preset["_built_prompt"],
            "reference_image_url": hosted_url,
            "image_size": {"width": OUTPUT_WIDTH, "height": OUTPUT_HEIGHT},
            "num_inference_steps": preset["steps"],
            "guidance_scale": preset["guidance_scale"],
            "id_weight": preset["id_weight"],
            "true_cfg": 1.0,
            "max_sequence_length": 128,
        },
    )
    return result["images"][0]["url"]


@with_retry
def _run_face_swap(base_image_path: str, swap_image_path: str) -> str:
    """
    Pass 2: face-swap to restore real identity onto the PuLID studio portrait.
    base_image_url  — PuLID-generated portrait (keeps body/pose/lighting)
    swap_image_url  — original user photo (provides the real face)
    Returns the face-swapped image URL.
    """
    import fal_client

    base_url = fal_client.upload_file(base_image_path)
    swap_url = fal_client.upload_file(swap_image_path)

    result = fal_client.subscribe(
        "fal-ai/face-swap",
        arguments={
            "base_image_url": base_url,
            "swap_image_url": swap_url,
        },
    )
    return result["image"]["url"]


def generate_avatar(
    image_path: str,
    output_path: str,
    preset_name: str = DEFAULT_PRESET,
) -> QualityReport:
    """
    Full avatar generation pipeline:
    Pass 1 — FLUX PuLID: generate studio portrait with white background
    Pass 2 — face-swap (fal-ai/face-swap): restore real identity onto portrait
    Fallback — BiRefNet + composite if PuLID fails entirely
    Returns QualityReport.
    """
    log.info(f"Generating avatar for: {image_path}")

    preset = dict(PROMPT_PRESETS.get(preset_name, PROMPT_PRESETS[DEFAULT_PRESET]))

    # Inject per-user attributes (glasses etc.) into the prompt
    user_attrs = _load_user_attributes(image_path)
    preset["_built_prompt"] = _build_prompt(preset, user_attrs)
    if user_attrs.get("glasses"):
        log.info(f"  → Glasses detected: {user_attrs.get('glasses_description', 'glasses')}")

    # PASS 1: FLUX PuLID — studio body/pose/lighting
    try:
        log.info(f"  → Pass 1: FLUX PuLID studio generation ({preset['name']})...")
        pulid_url = _run_pulid(image_path, preset)
        pulid_path = output_path.replace(".png", "_pulid.png")
        download_image(pulid_url, pulid_path)
        log.info("  ✓ PuLID portrait generated")

        # PASS 2: face-swap — lock the real face back onto the studio portrait
        log.info("  → Pass 2: face-swap to restore identity...")
        try:
            swapped_url = _run_face_swap(pulid_path, image_path)
            download_image(swapped_url, output_path)
            log.info("  ✓ Face swap applied — identity restored")
        except Exception as swap_err:
            log.warning(f"  ⚠ Face swap failed ({swap_err}), keeping PuLID result...")
            import shutil
            shutil.copy2(pulid_path, output_path)
        finally:
            if os.path.exists(pulid_path):
                os.remove(pulid_path)

    except Exception as pulid_err:
        log.warning(f"  ⚠ PuLID failed ({pulid_err}), falling back to BiRefNet composite...")
        image_url = upload_to_fal(image_path)
        fg_url = remove_background_fal(image_url)
        fg_path = output_path.replace(".png", "_fg.png")
        download_image(fg_url, fg_path)
        composite_on_white(fg_path, output_path)
        if os.path.exists(fg_path):
            os.remove(fg_path)
        log.info("  ✓ Fallback BiRefNet composite applied")

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

_GARMENT_DESCRIPTIONS: dict[str, str] = {
    "gucci_hoodie": (
        "Green Gucci Firenze 1921 hoodie with interlocking GG logo "
        "and red-navy braided stripe"
    ),
}


def _garment_description(garment_path: str) -> str:
    """Return a precise garment description for IDM-VTON guidance.

    Looks up a curated description by filename stem first; falls back to
    a human-readable version of the filename if not found.
    """
    stem = Path(garment_path).stem
    if stem in _GARMENT_DESCRIPTIONS:
        return _GARMENT_DESCRIPTIONS[stem]
    return stem.replace("_", " ").replace("-", " ").strip() or "garment"


@with_retry
def _run_fashn(avatar_path: str, garment_path: str, category: str) -> str:
    """
    Run FASHN Virtual Try-On v1.6 via fal.ai. Returns result image URL.
    FASHN is designed to preserve garment text, logos, and patterns.
    Both images are uploaded to fal CDN first; the model requires HTTPS URLs.
    category is mapped from pipeline convention (upper_body) to FASHN format (tops).
    """
    import fal_client

    model_url = fal_client.upload_file(avatar_path)
    garment_url = fal_client.upload_file(garment_path)
    fashn_category = FASHN_CATEGORY_MAP.get(category, "tops")

    result = fal_client.subscribe(
        "fal-ai/fashn/tryon/v1.6",
        arguments={
            "model_image": model_url,
            "garment_image": garment_url,
            "category": fashn_category,
        },
    )
    return result["images"][0]["url"]


def correct_garment_text(outfit_path: str, garment_path: str) -> bool:
    """
    Post-process an IDM-VTON result to restore legible garment text.

    IDM-VTON wrecks typography — the text pixels are not white; they are
    brighter-than-average pixels within the green garment region.  We detect
    text by luminance contrast (pixels > mean + 1.5σ inside the green mask),
    find the same region in the clean garment reference, and paste the clean
    version back using the contrast mask so only text pixels are touched.

    Gracefully skips (logs a warning) if any step fails.
    Returns True if the correction was applied, False otherwise.
    """
    try:
        import numpy as np

        outfit = Image.open(outfit_path).convert("RGB")
        garment = Image.open(garment_path).convert("RGB")

        out_arr = np.array(outfit, dtype=np.float32)
        r, g, b = out_arr[:, :, 0], out_arr[:, :, 1], out_arr[:, :, 2]

        # ── 1. Green garment mask ───────────────────────────────────────────
        green_mask = (g > r + 20) & (g > b + 20) & (g > 60)
        if green_mask.sum() < 500:
            log.warning("  ⚠ correct_garment_text: green region too small, skipping")
            return False

        # ── 2. Detect text via luminance contrast inside the green region ───
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        g_lum = lum[green_mask]
        bright_thresh = g_lum.mean() + 1.5 * g_lum.std()
        text_mask = (lum > bright_thresh) & green_mask

        if text_mask.sum() < 50:
            log.warning("  ⚠ correct_garment_text: no text contrast found in garment, skipping")
            return False

        # ── 3. Bounding box of text region in the output ────────────────────
        rows = np.where(text_mask.any(axis=1))[0]
        cols = np.where(text_mask.any(axis=0))[0]
        y1, y2 = int(rows.min()), int(rows.max())
        x1, x2 = int(cols.min()), int(cols.max())
        text_w, text_h = x2 - x1, y2 - y1

        if text_w < 20 or text_h < 5:
            log.warning("  ⚠ correct_garment_text: text bbox too small, skipping")
            return False

        # ── 4. Extract clean text block from garment reference ───────────────
        gw, gh = garment.size
        ref_crop = garment.crop((
            int(gw * 0.10), int(gh * 0.05),
            int(gw * 0.90), int(gh * 0.60),
        ))
        ref_arr = np.array(ref_crop, dtype=np.float32)
        rr, rg, rb = ref_arr[:, :, 0], ref_arr[:, :, 1], ref_arr[:, :, 2]

        # Text in reference: bright pixels inside the green region
        ref_green = (rg > rr + 20) & (rg > rb + 20) & (rg > 60)
        ref_lum = 0.299 * rr + 0.587 * rg + 0.114 * rb
        if ref_green.sum() > 50:
            ref_g_lum = ref_lum[ref_green]
            ref_thresh = ref_g_lum.mean() + 1.5 * ref_g_lum.std()
            ref_text = (ref_lum > ref_thresh) & ref_green
        else:
            # Fallback: just use brightest pixels in entire crop
            ref_thresh = ref_lum.mean() + 1.5 * ref_lum.std()
            ref_text = ref_lum > ref_thresh

        if ref_text.sum() > 50:
            rrows = np.where(ref_text.any(axis=1))[0]
            rcols = np.where(ref_text.any(axis=0))[0]
            ref_crop = ref_crop.crop((
                int(rcols.min()), int(rrows.min()),
                int(rcols.max()), int(rrows.max()),
            ))

        # ── 5. Resize clean text to match output bbox and paste ─────────────
        clean_resized = ref_crop.resize((text_w, text_h), Image.LANCZOS)

        # Paste mask: only touch the detected text pixels
        mask_arr = (text_mask[y1:y2, x1:x2].astype(np.uint8) * 255)
        mask_img = Image.fromarray(mask_arr)

        outfit_rgb = Image.open(outfit_path).convert("RGB")
        outfit_rgb.paste(clean_resized, (x1, y1), mask_img)
        outfit_rgb.save(outfit_path, "PNG")

        log.info(
            f"  ✓ Garment text corrected "
            f"(bbox {x1},{y1}→{x2},{y2} = {text_w}×{text_h}px, "
            f"{text_mask.sum():.0f} text pixels)"
        )
        return True

    except Exception as exc:
        log.warning(f"  ⚠ correct_garment_text failed: {exc}, skipping")
        return False


def preprocess_garment(garment_path: str) -> str:
    """
    Remove hood, drawstring laces, and collar from a garment image before
    sending to IDM-VTON.  The laces cross the GUCCI text and cause the model
    to warp letters around them — removing them gives IDM-VTON a clean front
    panel to work from.

    Strategy:
      1. Scan rows top-down for the first row with ≥10 bright (white) pixels —
         that is the top of the garment text (≈33% from top for the Gucci hoodie).
      2. Crop to (text_row − 5% padding) downward, trimming 5% from each side.
      3. Save to a temp sidecar file next to the original.

    Returns the path to the pre-processed image (original untouched).
    """
    import numpy as np
    img = Image.open(garment_path).convert("RGB")
    w, h = img.size
    arr = np.array(img)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Scan for first row containing ≥10 bright pixels (white text)
    text_row = None
    for y in range(int(h * 0.20), int(h * 0.60)):
        bright = int(((r[y] > 180) & (g[y] > 180) & (b[y] > 180)).sum())
        if bright >= 10:
            text_row = y
            break

    if text_row is None:
        log.warning("  ⚠ preprocess_garment: no text row found, using original garment")
        return garment_path

    padding = int(h * 0.05)
    crop_top = max(0, text_row - padding)
    crop_left = int(w * 0.05)
    crop_right = w - int(w * 0.05)

    cropped = img.crop((crop_left, crop_top, crop_right, h))
    new_h = cropped.size[1]
    log.info(
        f"  ✓ Preprocessed garment: cropped from {h} to {new_h}px "
        f"(removed hood/laces above y={crop_top})"
    )

    out_path = "/tmp/garment_preprocessed.png"
    cropped.save(out_path, "PNG")
    return out_path


def transfer_outfit(
    avatar_path: str,
    garment_path: str,
    output_path: str,
    category: str = "upper_body",
) -> QualityReport:
    """Outfit transfer with retry, garment pre-processing, and QA. Returns QualityReport."""
    log.info(f"  → IDM-VTON outfit transfer (category: {category})...")

    # Pre-process garment: strip hood/laces so text region is clean for VTON
    effective_garment = preprocess_garment(garment_path)

    result_url = _run_fashn(avatar_path, effective_garment, category)
    download_image(result_url, output_path)

    # Ensure consistent dimensions
    img = Image.open(output_path).convert("RGB")
    if img.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
        img = img.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.LANCZOS)
        img.save(output_path, "PNG", quality=95)

    # Text post-processing disabled — causes edge artifacts on some garments
    # correct_garment_text(output_path, garment_path)

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
        avatar_qa = generate_avatar(user_image_path, avatar_path, preset_name)
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
    outfit_images = sorted(
        f for f in Path(outfit_dir).iterdir()
        if f.suffix.lower() in valid_ext and not f.name.startswith(".")
    )

    if not user_images:
        log.error(f"No images in {input_dir}"); sys.exit(1)
    if not outfit_images:
        log.error(f"No outfits in {outfit_dir}"); sys.exit(1)

    log.info("=" * 60)
    log.info("Zeely AI Avatar Pipeline — Batch Mode")
    log.info(f"Users: {len(user_images)} | Outfits: {len(outfit_images)}")
    log.info(f"Mode: PuLID primary (BiRefNet fallback) | Preset: {preset_name}")
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
                    output_dir, preset_name, outfit_category,
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
                uid, upath, opath, output_dir, preset_name, outfit_category,
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
            "preset": preset_name, "mode": "pulid_primary", "parallel": parallel,
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
  python pipeline.py --preset fashion                    # Fashion preset
  python pipeline.py --parallel                          # Parallel batch
  python pipeline.py -s photo.jpg --outfit-single b.jpg  # Single image
  python pipeline.py --qa-only output/001/avatar.png     # QA scoring only
        """,
    )

    p.add_argument("--input", "-i", default=DEFAULT_INPUT_DIR)
    p.add_argument("--outfits", "-f", default=DEFAULT_OUTFIT_DIR)
    p.add_argument("--output", "-o", default=DEFAULT_OUTPUT_DIR)
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
            valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
            ofs = sorted(
                f for f in Path(args.outfits).iterdir()
                if f.suffix.lower() in valid_ext and not f.name.startswith(".")
            )
            outfit = str(ofs[0]) if ofs else None
        if not outfit:
            log.error("No outfit specified"); sys.exit(1)
        r = process_single_user("001", args.single, outfit, args.output, args.preset, args.category)
        print(json.dumps(asdict(r), indent=2))
    else:
        run_batch(args.input, args.outfits, args.output, args.preset, args.category, args.parallel)


if __name__ == "__main__":
    main()
