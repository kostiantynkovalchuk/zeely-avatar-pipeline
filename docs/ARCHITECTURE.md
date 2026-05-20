# Pipeline Design Decisions

Technical rationale behind every major choice in the pipeline, including tradeoffs, failure modes, and scaling considerations.

---

## Why BiRefNet (not rembg, U2Net, or SAM)

**Choice**: fal.ai BiRefNet v2 with `BiRefNet-portrait` model variant.

**Rationale**: The task requires pixel-perfect separation on pure white (#FFFFFF). Hair strands, semi-transparent fabric edges, and earrings are the hardest cases for background removal — they create visible haloing artifacts on white backgrounds that ruin the studio photo illusion.

BiRefNet's portrait-specific model was trained on portrait segmentation datasets with emphasis on fine boundary detail. The `refine_foreground` option adds an additional matting pass that cleans alpha edges. In comparative testing, it handles hair significantly better than U2Net-based rembg, and it runs as a managed API (no GPU infrastructure to manage).

**Tradeoff**: More expensive than self-hosted rembg (~$0.02 vs ~$0.001), but the quality difference on hair edges justifies the cost when the output must look like a real studio photo.

**Fallback**: If BiRefNet API is unavailable, the pipeline's retry logic will attempt 3 times with backoff. A future improvement could add rembg as a local fallback: lower quality but zero API dependency.

---

## Why FLUX PuLID (not InstantID, FaceID, or PhotoMaker)

**Choice**: fal.ai FLUX PuLID for optional identity-preserving studio relighting.

**Rationale**: The core problem PuLID solves is *lighting inconsistency*. A user photo taken outdoors with harsh sunlight produces a very different look than a studio softbox setup, even after background removal. PuLID allows re-rendering the scene with studio lighting while preserving the person's identity through face embeddings.

Why PuLID over alternatives:
- **InstantID** scores higher on raw face similarity but tends toward "over-copying" the reference — it can reproduce background noise and lighting artifacts from the source. PuLID's lighter identity conditioning allows the diffusion model to properly re-render the lighting.
- **IP-Adapter FaceID** is more flexible (works with any SD model) but weaker at preserving fine identity details like moles, skin texture, and hair part. For an avatar product, every recognizable detail matters.
- **PhotoMaker** requires multiple reference images for best results. This pipeline needs to work with a single input photo.

**Tradeoff**: PuLID adds ~$0.06/image and ~10-15 seconds. It's optional (`--pulid` flag) because for well-lit source photos, BiRefNet + white composite produces good results without it.

**Known failure mode**: PuLID can struggle with extreme head angles (>30° off-center) and heavy occlusion (sunglasses, masks). The pipeline spec requires "neutral, frontal" pose, which is PuLID's strongest scenario.

---

## Why IDM-VTON (not OOTDiffusion, CatVTON, or StableVITON)

**Choice**: Replicate-hosted IDM-VTON (ECCV 2024).

**Rationale**: Virtual try-on quality depends on two things: garment texture fidelity and identity preservation during the transfer. IDM-VTON outperforms alternatives on both metrics because of its architectural approach — it directly fine-tunes the diffusion UNet for the try-on task, rather than conditioning from a side network.

Comparison:
- **OOTDiffusion**: Good garment texture but weaker at preserving face identity. The model sometimes "bleeds" garment patterns into skin.
- **CatVTON** (with FLUX Fill): Newer, promising results on benchmarks but slower (~35s) and less stable on diverse body types. Running FLUX-based fill inside a try-on loop is compute-heavy.
- **StableVITON**: Based on SD 1.5 (frozen backbone + ControlNet). The 512px native resolution limits output quality for the 768×1024 target.

IDM-VTON on Replicate gives ~19s inference on A100, auto-mask generation (no manual masking needed), and stable results across body types.

**Tradeoff**: CC BY-NC-SA 4.0 license — non-commercial only. For production deployment, Zeely would need either a commercial license arrangement or an alternative commercial model. This is appropriate for a test task / proof of concept.

**Known failure mode**: IDM-VTON can distort hands when the garment reference image includes visible hands. The auto-mask sometimes leaks into skin regions on tight-fitting garments. Mitigation: post-processing QA checks flag these issues.

---

## Quality scoring: what it catches and what it misses

**What the automated scorer catches**:
- Blank/failed outputs (bg_purity = 0%, no face detected)
- AI blur / "plastic skin" (Laplacian < 80)
- Incorrect dimensions from API size drift
- Background contamination (non-white border pixels)

**What it currently misses** (future improvements):
- **Identity drift**: A face similarity score (ArcFace / insightface embeddings comparing source → avatar) would catch cases where PuLID generates a similar but noticeably different person. This is the single most impactful QA addition.
- **CLIP similarity**: Comparing the garment reference against the outfit output using CLIP embeddings would catch cases where IDM-VTON produces a different garment color or type.
- **Anatomical artifact detection**: Extra fingers, merged hands, asymmetric ears. Would require a specialized detector or a vision-language model pass.
- **Aesthetic scoring**: LAION aesthetic predictor to filter out technically correct but visually unappealing outputs.

The current scorer is lightweight and runs locally with zero API cost. It's designed as a triage layer — it catches the obvious failures and flags edge cases for human review.

---

## Product fidelity risks at scale

When this pipeline runs for thousands of users, these failure patterns would emerge:

**1. Identity preservation degradation on edge cases**
- Very dark or very light skin tones: BiRefNet's edge detection and PuLID's skin rendering both perform best on medium skin tones. A production system needs diverse test sets.
- Non-standard hair: Very curly hair, braids, hijabs, bald heads — each requires different edge handling thresholds. BiRefNet's portrait model is reasonable but not perfect.
- Accessories: Glasses frames often get partial removal. Earrings can disappear or double.

**2. Outfit transfer instability**
- Pattern distortion: Stripes and geometric patterns can warp during try-on. This is a fundamental limitation of current diffusion-based try-on.
- Skin-tone mixing: On tight garments, the auto-mask sometimes includes skin, causing the model to repaint skin in garment texture.
- Category mismatch: A blazer reference image passed with `category=dresses` will produce corrupted output. The pipeline currently trusts the user to specify correctly.

**3. API reliability at volume**
- Both fal.ai and Replicate have request rate limits and cold-start latency. The retry + backoff + parallel (bounded) design handles this, but at 10,000+ images/day, a queue-based architecture would be needed.

---

## Scalability considerations

**Current state** (this submission): Sequential or 3-worker parallel, both synchronous.

**Next level** (production MVP):
- **Job queue** (Redis + Celery or similar): Decouple submission from processing. Users submit photos, workers process async, webhook on completion.
- **Result caching**: If the same user photo is processed twice (re-run, different outfit), skip avatar generation and reuse the cached avatar.
- **CDN for outputs**: Currently saves to local filesystem. Production needs S3/GCS with signed URLs.

**Scale target** (10K+ users/day):
- **Kubernetes workers** with auto-scaling based on queue depth
- **Model hosting migration**: Move IDM-VTON from Replicate to self-hosted (Replicate's per-inference pricing is expensive at volume; a dedicated A100 instance costs ~$2/hr and processes ~200 images/hr ≈ $0.01/image vs $0.03)
- **Batch API endpoints**: Both fal.ai and Replicate offer batch/async APIs that are cheaper than synchronous calls
- **Quality-gated pipeline**: Auto-QA → auto-retry with different seed if failed → human review queue for persistent failures

---

## Why this specific API combination

The pipeline deliberately splits work across two API providers rather than building everything on one:

**fal.ai** handles image processing (bg removal, PuLID) because:
- Sub-second latency on BiRefNet
- FLUX model family support (PuLID requires FLUX backbone)
- Upload CDN included (no separate storage needed)

**Replicate** handles IDM-VTON because:
- IDM-VTON is hosted there with optimized inference (A100 80GB)
- The model isn't available on fal.ai
- Replicate's prediction API is well-suited for longer inference jobs (19s)

A production system could consolidate onto one provider by self-hosting IDM-VTON, but for rapid prototyping and testing, using each model where it's best hosted makes sense.
