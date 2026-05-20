# Zeely AI Avatar Generator Pipeline

Automated two-stage engine: **user photo → studio avatar on white background → outfit transfer**.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API keys
export FAL_KEY="your-key"            # https://fal.ai/dashboard/keys
export REPLICATE_API_TOKEN="your-key" # https://replicate.com/account/api-tokens

# 3. Add images
#    input/users/   ← user photos (JPG/PNG/WebP)
#    input/outfits/ ← garment reference images

# 4. Run
python pipeline.py                   # Basic batch
python pipeline.py --pulid           # + studio relight
python pipeline.py --parallel        # Parallel processing
python pipeline.py --preset fashion  # Fashion editorial preset
```

---

## Architecture

```
User Photo ──→ [BiRefNet Portrait] ──→ [White BG Composite] ──→ avatar.png
                                                                    │
Garment Ref ──────────────────────────────→ [IDM-VTON] ──→ avatar_outfit.png
                                                                    │
                                                              [Quality Scoring]
                                                                    │
                                                            batch_report.json
```

### Stage 1 — Avatar generation

| Step | Tool | What it does |
|------|------|------|
| 1a | fal.ai BiRefNet v2 (portrait) | Background removal with hair/edge refinement |
| 1b | Pillow compositing | White #FFFFFF background, auto-crop to 3:4, resize to 768×1024 |
| 1c | fal.ai FLUX PuLID *(optional)* | Identity-preserving studio relight via face embeddings |
| 1d | Quality scorer | Background purity, sharpness, face detection, dimensions |

### Stage 2 — Outfit transfer

| Step | Tool | What it does |
|------|------|------|
| 2a | Replicate IDM-VTON | Virtual try-on with auto-masking |
| 2b | Post-processing | Dimension normalization |
| 2c | Quality scorer | Same metrics as stage 1 |

---

## Production features

### Retry with exponential backoff

Every API call wraps in `@with_retry` — handles network drops, rate limits (429), and server errors (5xx) with 2s → 4s → 8s backoff over 3 attempts.

### Upload caching

`UploadCache` hashes file content (MD5) before uploading. Same image referenced twice → one CDN upload. In batch processing with shared outfits, this eliminates redundant transfers.

### Quality scoring

`assess_quality()` runs four automated checks per image:

| Metric | Method | Threshold |
|--------|--------|-----------|
| Background purity | Border pixel sampling (outer 5%) | ≥ 95% near-white |
| Sharpness | Laplacian variance on center crop | ≥ 80 |
| Face presence | Skin-tone heuristic in upper-center | ≥ 10% samples |
| Dimensions | Exact match to 768×1024 | Exact |

Results are embedded in the batch report JSON for every image.

### Parallel processing

`--parallel` flag enables `ThreadPoolExecutor` with bounded concurrency (default 3 workers). Respects API rate limits while processing multiple users simultaneously.

### Configurable prompt presets

Three built-in presets with different guidance scales, identity weights, and step counts:

| Preset | Use case | Steps | Guidance | ID weight |
|--------|----------|-------|----------|-----------|
| `studio` | Clean headshots | 30 | 4.0 | 1.0 |
| `fashion` | Editorial look | 35 | 5.0 | 0.9 |
| `ecommerce` | Product shots | 25 | 3.5 | 1.0 |

---

## CLI reference

```bash
python pipeline.py [options]

Options:
  -i, --input DIR        User photos directory (default: input/users)
  -f, --outfits DIR      Outfit references directory (default: input/outfits)
  -o, --output DIR       Output directory (default: output)
  --pulid                Enable PuLID studio enhancement
  -p, --preset NAME      Prompt preset: studio|fashion|ecommerce
  -c, --category CAT     IDM-VTON category: upper_body|lower_body|dresses
  --parallel             Enable parallel batch processing
  -s, --single PATH      Process single image instead of batch
  --outfit-single PATH   Outfit image for single mode
  --qa-only PATH         Score an existing image without generating
```

---

## Output structure

```
output/
├── 001/
│   ├── avatar.png            # 768×1024, white BG
│   └── avatar_outfit.png     # Same person, new outfit
├── 002/
│   ├── avatar.png
│   └── avatar_outfit.png
├── 003/ ...
└── batch_report.json          # Full results with QA and timings
```

### Batch report sample

```json
{
  "summary": {
    "total": 3,
    "success": 3,
    "qa_passed": 2,
    "failed": 0,
    "seconds": 87.3,
    "avg_per_user": 29.1
  },
  "results": [
    {
      "user_id": "001",
      "status": "success",
      "avatar_qa": {
        "bg_purity": 0.98,
        "blur_score": 142.5,
        "face_detected": true,
        "passed": true
      },
      "timings": { "avatar_s": 12.3, "outfit_s": 18.1, "total_s": 30.4 }
    }
  ]
}
```

---

## Cost per user

| API call | Cost |
|---|---|
| fal.ai BiRefNet v2 | ~$0.02 |
| fal.ai FLUX PuLID *(optional)* | ~$0.06 |
| Replicate IDM-VTON | ~$0.03 |
| **Total (without PuLID)** | **~$0.05** |
| **Total (with PuLID)** | **~$0.11** |

---

## Extending the pipeline

### Text-to-garment generation

When no garment reference image is available:

```python
import fal_client
result = fal_client.subscribe("fal-ai/flux/dev", arguments={
    "prompt": "Navy blue blazer, product flat lay, white background",
    "image_size": {"width": 768, "height": 1024},
})
# Feed result into IDM-VTON step
```

### Multiple outfits per user

The pipeline auto-cycles through `input/outfits/`. For N outfits × M users, run N passes with different outfit directories.

---

## Dependencies

```
fal-client>=0.5.0
replicate>=1.0.0
Pillow>=10.0.0
requests>=2.31.0
```

Python 3.10+. No GPU required (all inference is cloud API).
