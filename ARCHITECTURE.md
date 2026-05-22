# Pipeline Design Decisions

Technical narrative of how and why the Zeely AI Avatar Pipeline was built and iterated.

---

## Evolution of the Approach

### Phase 1: Background Removal + Compositing (BiRefNet)

**Initial approach:** remove background with fal.ai BiRefNet → composite person on white → IDM-VTON for outfit transfer.

**Result:** cutout quality, not studio quality. The composited images looked like product catalog photos with pasted-in people — visible edge artifacts, inconsistent lighting, flat. Compared to Zeely's reference outputs (full AI-generated studio portraits), this approach was fundamentally limited in what it could produce.

**Decision:** pivot to full portrait generation. Instead of removing and compositing, generate the entire studio portrait from scratch using the user photo as an identity reference.

---

### Phase 2: PuLID as Primary Generator

**FLUX PuLID** generates the entire studio portrait — background, lighting, pose, framing — from a single user photo as identity reference, guided by a detailed text prompt.

**Solved:**
- Studio lighting, consistent white background, professional quality
- Controllable pose (arms down, half-body framing) via prompt
- Body type and glasses injected via `user_attributes.json` → PuLID prompt

**New problems:**
- Identity drift — PuLID sometimes changes hair color, removes glasses, alters facial features
- Body proportions require manual specification per user

**BiRefNet fallback retained** in `generate_avatar()` for cases where PuLID fails entirely (network errors, API timeouts).

---

### Phase 3: Two-Pass Identity Restoration (Face-Swap)

**Two-pass approach:**
1. PuLID generates body, pose, lighting, and studio environment
2. `fal-ai/face-swap` pins the real face from the source photo back onto the PuLID portrait

**Solved:** glasses preserved, skin texture locked, facial hair and moles retained.

**Per-user customization via `user_attributes.json`:**
- `body_description` — injected into PuLID prompt; controls gender, build, hair color
- `glasses` + `glasses_description` — injected into prompt so PuLID generates glasses as part of the portrait (face-swap alone would lose them)

Prompt injection order: body description first, then glasses, producing natural language: *"this slim young woman with narrow shoulders wearing round pink-tinted glasses"*.

---

### Phase 4: FASHN v1.6 for Garment Text Preservation

**IDM-VTON problem:** UNet-based diffusion warped garment typography — `GUCCI` → `GUCCCI`, `FIRENZE` → `TRENZE`. The model operates in latent space and cannot guarantee pixel-level text fidelity.

**Research:** TED-VITON addresses text preservation but is academic-only, not hosted. FASHN v1.6 is specifically designed for pixel-space garment rendering and is available on fal.ai.

**FASHN v1.6 advantages over IDM-VTON:**
- Operates in pixel space → text, logos, stitching details preserved
- Maskless inference — no manual garment segmentation required
- Faster: ~11s vs ~25s per transfer
- Higher native resolution (864×1296)

**Consolidation:** moving from IDM-VTON on Replicate to FASHN v1.6 on fal.ai eliminated the second API provider entirely. The full pipeline now runs on fal.ai with a single API key and billing dashboard.

---

## Why Each Model Was Chosen

### FLUX PuLID — Avatar Generation

Preserves facial identity through face embeddings while allowing full body and scene generation. The key capability is that it generates a *new* image — correct studio lighting, white background, controlled pose — while maintaining recognizable identity from the reference photo.

**Alternative considered:** InstantID — stronger identity lock but less generation freedom, tends to reproduce the exact pose and framing from the source photo rather than generating a clean studio portrait.

### fal-ai/face-swap — Identity Restoration

Lightweight refinement pass (~3s) that solves PuLID's identity drift without replacing PuLID's body/pose generation. The face-swap preserves glasses, skin texture, facial hair, and subtle facial features that PuLID generalizes away.

Runs as Pass 2, not a replacement for Pass 1. If face-swap fails, the pipeline falls back to the raw PuLID result.

### FASHN v1.6 — Outfit Transfer

Pixel-space virtual try-on model designed specifically for garment detail preservation. Handles text, logos, patterns, and stitching. Maskless inference simplifies the call — only `model_image`, `garment_image`, and `category` (tops/bottoms/one-piece) are required.

---

## User Attributes System

`input/users/user_attributes.json` provides per-user customization without code changes. The pipeline loads attributes by source filename and injects them into the PuLID prompt before generation.

**Extensible:** adding `skin_tone`, `pose_preference`, or `hair_color` as new fields requires only a JSON edit and a one-line change in `_build_prompt()`.

**Glasses detection could be automated** in production using a face analysis model (e.g., insightface attribute prediction) to detect whether the source photo shows glasses and extract their description.

---

## Known Limitations

**Garment text fidelity:** FASHN v1.6 preserves text significantly better than IDM-VTON but minor spacing artifacts remain on some outputs (`G UCCI` vs `GUCCI`). This is a fundamental diffusion model limitation — pixel-space models do not have a text rendering module.

**Face-swap neck boundary:** on some body types, face-swap creates a faint seam at the neck/shoulder boundary. Most noticeable on users with very different skin tones between face and neck.

**Accessories not supported:** FASHN handles garments only (tops, bottoms, one-piece). Hats, sunglasses, shoes, and jewelry are outside the model's scope. Accessories would require region-specific inpainting, e.g., FLUX Fill targeting the head region for hats.

**Pose compliance:** PuLID occasionally ignores pose instructions (arms crossed, hand in pocket) despite explicit negative prompts. Resolved by retry — each PuLID call uses a different seed.

**Body proportions:** must be manually specified per user in `user_attributes.json`. Could be automated with a body estimation model (e.g., DWPose, OpenPose) that extracts build descriptors from the source photo.

---

## Supplementary Experiments

These were run against the same four users to test model boundaries. Results saved in `experiments/`.

**Hawaiian shirt (Polo Ralph Lauren tropical floral):** 4/4 users succeeded. Complex floral pattern preserved well by FASHN v1.6.

**Cowboy hat:** FASHN treated the hat as a garment and attempted to wrap it around the torso. Confirmed: VTON models cannot handle accessories. Headwear requires a different inference approach (inpainting on head region).

**Nike sneakers:** avatar is half-body framed to hip level — footwear is not visible in the output at all. Would require a full-body pipeline extension with different framing.

**Denim jacket (structured garment):** button and hardware detail is not preserved. FASHN v1.6 renders the fabric and color well but small metal details (rivets, buttons) are approximated. Larger body types show more visible face-swap boundary artifacts at the collar.

---

## Scalability Considerations

**Current:** sequential processing, ~55s per user end-to-end.

**Production path:**
- Parallel processing: `--parallel` flag with `ThreadPoolExecutor` (3 workers, rate-limit safe)
- Job queue: Redis + Celery for async user submission
- FASHN batch API: reduces per-call overhead at high volume
- Self-hosted PuLID: cost optimization at >1000 users/day

**Quality gate:** automated QA → retry with different seed → human review queue for persistent failures. Current QA thresholds (95% border purity, Laplacian variance ≥80) are conservative; real production thresholds should be tuned on labeled data.

---

## Infrastructure: Single Provider (fal.ai)

**Started with:** fal.ai (PuLID, face-swap, BiRefNet) + Replicate (IDM-VTON). Two providers, two API keys, two billing dashboards, two retry behaviors.

**Replicate issues:** cold-start reliability — IDM-VTON models occasionally stuck in "starting" state for 60–90s, sometimes timing out entirely.

**Consolidated to fal.ai only:** all three pipeline stages available natively. Single API key (`FAL_KEY`), consistent `fal_client.subscribe()` call pattern across all models, single billing source.
