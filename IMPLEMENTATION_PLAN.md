# Image Identity Classifier — Implementation Plan

## 1. Goal

Sort the images in an input directory into folders, one folder per reference image, by **identity**: an image goes into `ref_k/` when it shows the *same* object or subject as reference image `k`. It does not go there when it merely shows the same *kind* of thing.

| | Description |
|---|---|
| **Input 1** | A directory of candidate images (N images) |
| **Input 2** | 1–5 reference images (M ≤ 5) |
| **Output** | `out/<ref_name>/` per reference, plus `out/_unmatched/` and `out/_review/`, and a `report.csv` |

### Approach

The pipeline has two stages.

1. **Local embeddings (every image).** DINOv2 embeds every image locally, and cosine similarity is computed against each reference. Clear matches and clear non-matches are decided here, at zero API cost.
2. **Jev (borderline only).** Candidates whose similarity falls in an uncertain band go to TypeSafe Jev in **batched requests**. Each request carries the reference images **once** in shared `state`, plus a batch of candidates.

## 2. Requirements

### Runtime

- Python 3.11+.
- **Local embedding model:** DINOv2 (`facebook/dinov2-base` by default) runs through PyTorch and Hugging Face `transformers`. The weights download on first run (about 350 MB) and are cached locally.
  - **GPU:** optional. CPU is fine for hundreds to low thousands of images. A CUDA GPU is recommended for larger sets.
  - **Alternative:** SigLIP (`google/siglip-base-patch16-224`) can be selected by config. DINOv2 is the default because its features capture instance-level detail better, while SigLIP/CLIP features are tuned toward semantic, text-aligned similarity. Calibration (§6) decides.
- **Ollama is not required, and as far as I know it can't do this job.** Ollama's embedding endpoint (`/api/embed`) accepts text input only, and its model library does not ship DINOv2 or SigLIP as image-embedding models. Ollama can run vision *chat* models such as LLaVA, but those generate text and do not return image vectors for cosine similarity. The plan therefore runs DINOv2/SigLIP in-process with PyTorch, which is also fully local and offline after the first download. If Ollama is mandated for operational reasons, verify current image-embedding support in Ollama before committing; this is the one requirement to re-check.

### Python packages

| Package | Purpose |
|---|---|
| `torch`, `transformers` | DINOv2 / SigLIP inference |
| `pillow` (+ `pillow-heif` if HEIC input is expected) | Decoding, EXIF orientation, resizing, JPEG re-encoding |
| `numpy` | Vectors, cosine similarity |
| `typesafe` (Python SDK) or `httpx` | Jev API calls. Use the SDK if its image object supports `assets`, otherwise raw HTTP multipart. |
| `typer` | CLI |
| `pytest` | Tests |

### TypeSafe Jev

- A TypeSafe API key, set in the `TYPESAFE_API_KEY` environment variable and never committed. The direct API is waitlisted (early access).
- If a gateway (Vercel AI Gateway, OpenRouter, LiteLLM) is used instead, first confirm it forwards the `TypeSafe-Danger-Zone-Images: true` header and `assets`.
- A vision-capable model: `jev-1.14` (or 1.13).
- Every request that contains images must send the `TypeSafe-Danger-Zone-Images: true` header. Long context, if needed, also requires `TypeSafe-Danger-Zone-Long-Context: true`.
- **Status:** TypeSafe marks image input as experimental, and formats, limits and accuracy may change. Jev is used only for the borderline band, so the app still works with it disabled (`--no-jev` routes borderline images to `_review/`).

### Jev image limits (from docs.typesafe.ai Images page, read 2026-10-20)

| Limit | Value |
|---|---|
| Format | PNG or JPEG, still images only |
| Size per image | ≤ 16 MiB |
| Width and height | 280–1568 px each |
| Total pixels | 78,400–2,458,624 |
| Request body | ≤ 32 MiB, including base64 and multipart framing |
| Expanded request | ≤ 32 MiB after every reference is substituted; each *use* of a reference adds a copy |
| Validation | Out-of-range images are **rejected, not resized**, and one bad image fails the **whole** request |
| Assets | Every asset must be referenced, otherwise the request returns 400 |

## 3. Architecture

```
inputs ──► [1 Scan & validate] ──► [2 Preprocess] ──► [3 Embed (DINOv2, cached)]
                                                            │
                                                  [4 Similarity + banding]
                                     ┌──────────────┬───────┴──────────┐
                                  ACCEPT         BORDERLINE          REJECT
                                     │              │                   │
                                     │     [5 Jev batch judge]          │
                                     │      ┌───────┼────────┐          │
                                     ▼      ▼       ▼        ▼          ▼
                                  out/ref_k/   out/_review/      out/_unmatched/
                                                    │
                                         [6 report.csv + summary]
```

### Module layout

```
image_classifier/
  cli.py            # typer entry point
  config.py         # thresholds, model names, batch limits (TOML file + CLI overrides)
  io_scan.py        # discovery, validation, hashing
  preprocess.py     # EXIF fix, RGB, embedding tensor, Jev-compliant JPEG
  embed.py          # DINOv2/SigLIP wrapper + on-disk cache
  similarity.py     # cosine matrix, banding decision
  jev_judge.py      # batch packing, request building, response parsing, retries
  route.py          # copy files, collision-safe naming
  report.py         # report.csv + console summary
tests/
calibrate.py        # threshold sweep on a labeled set
```

## 4. Pipeline detail

### Step 1: Scan and validate

- Recurse into input 1 (configurable) and accept `.jpg .jpeg .png .webp .bmp .tif .heic`. Skip unreadable files and log them in the report as `error`.
- Input 2 must contain 1–5 images, otherwise exit with an error. Folder names come from the reference filename stems, sanitized and deduplicated.
- Hash each file (SHA-256) to key the embedding cache and to detect duplicates. A reference image that also appears in input 1 is routed directly to its own folder.

### Step 2: Preprocess

The preprocessor produces two derivatives per image, both in memory or in a temp cache:

- **Embedding input:** apply EXIF transpose, convert to RGB, then apply the model's own processor (resize and normalize).
- **Jev input** (built lazily, only for borderline candidates and references): apply EXIF transpose and convert to RGB. Downscale so the longest side is ≤ `jev.max_side` (default 1024, which stays inside 1568 and keeps payloads small). Upscale if either side is < 280 px, and enforce the total-pixel bounds. Re-encode as JPEG at quality 90. Validate against §2 limits **before** sending, because one bad image would fail the whole batch.

### Step 3: Embed

- DINOv2: take the CLS token, optionally concatenated with mean-pooled patch tokens (config flag, chosen by calibration), then L2-normalize.
- **Reference robustness:** embed each reference plus its horizontal flip and average them, which slightly reduces pose sensitivity. More reference augmentation is a later option.
- Cache vectors in `.cache/embeddings/<model>/<sha256>.npy`. Re-runs and threshold changes then need no recomputation.
- Run inference in batches (default 16) under `torch.inference_mode()`, using the GPU if available.

### Step 4: Similarity and banding

For each candidate, compute the cosine similarity to each reference: `s = C · Rᵀ`, an N×M matrix. Let `s1` be the best score (for reference `r1`) and `s2` the second best (0 if M = 1).

| Band | Rule | Action |
|---|---|---|
| **ACCEPT** | `s1 ≥ T_high` and `s1 − s2 ≥ margin` | Copy to `out/<r1>/` |
| **REJECT** | `s1 < T_low` | Copy to `out/_unmatched/` |
| **BORDERLINE** | Everything else, including high similarity to two references at once | Send to the Jev judge |

The defaults are placeholders until calibration (§6): `T_high = 0.70`, `T_low = 0.45`, `margin = 0.05`. Raw DINOv2 cosine values differ by domain, so these numbers must not be trusted unchanged.

### Step 5: Jev batch judge

This is the cost-critical design. **Reference images are sent once per request in shared `state`, never once per candidate.** The Images docs specify that an image in `state` is evaluated against every question in the request. An image placed in a question's `instructions` or `criteria`, by contrast, is duplicated for each question that uses it.

#### Request shape (one request per batch of K borderline candidates)

```jsonc
// POST /v1/systemone, header TypeSafe-Danger-Zone-Images: true
{
  "model": "jev-1.14",
  "state": {
    "task": "Identity matching. References show specific individual objects/subjects. Candidates are photos that may or may not show one of those same individuals.",
    "references": {
      "ref_1": { "name": "red_bike",   "image": "<img:ref_1>" },
      "ref_2": { "name": "blue_bike",  "image": "<img:ref_2>" }
      // ... up to ref_5
    },
    "candidates": {
      "c_0": { "image": "<img:c_0>" },
      "c_1": { "image": "<img:c_1>" }
      // ... up to c_{K-1}
    }
  },
  "questions": {
    "c_0": {
      "type": "choice",
      "instructions": "Which reference shows the very same individual object/subject as `candidates.c_0.image`? Same individual means identical identity (e.g. the same specific item, not merely the same model, brand, breed or category). Ignore differences in angle, lighting, background, crop and scale.",
      "criteria": {
        "ref_1": "Same individual as `references.ref_1.image`",
        "ref_2": "Same individual as `references.ref_2.image`",
        "none":  "Not the same individual as any reference (may be similar-looking or the same type)"
      }
    },
    "c_1": { "...": "same shape, pointing at `candidates.c_1.image`" }
  },
  "assets": {
    "<img:ref_1>": { "type": "image", "format": "jpeg", "data_base64": "..." },
    "<img:c_0>":   { "type": "image", "format": "jpeg", "data_base64": "..." }
    // every asset referenced exactly once, in state
  }
}
```

The criteria and instructions point at images **by state path** (backticked), not by asset placeholder. As a result, no image is duplicated across questions. Each image appears once in the payload. Prefer multipart upload over base64 for about 25% smaller bodies.

#### Batch packing

- Keep K at or below `jev.max_batch` (default 16). Also keep the body at or below `jev.max_request_mib` (default 24, leaving headroom under 32).
- Pack greedily by encoded byte size, after reserving space for the M references.
- A `--max-jev-requests` cap guards against surprise spend.

#### Decision

Each answer returns `choice`, `confidence` and `probabilities`.

| Condition | Action |
|---|---|
| `choice = ref_k` and `probabilities[ref_k] ≥ J_accept` (placeholder 0.80) | `out/ref_k/` |
| `choice = none` and `probabilities[none] ≥ J_accept` | `out/_unmatched/` |
| Otherwise | `out/_review/` (a human decides) |

#### Failure handling

| Failure | Response |
|---|---|
| 422 or 400 | Log validation errors. Bisect the batch to isolate the bad image, then retry the remainder. |
| 429 or 5xx | Exponential backoff with jitter, up to 3 retries. |
| Persistent failure, or `--no-jev` | Route the batch to `_review/`; never guess. |
| Every response | Record `usage.input_tokens` in the report for cost tracking. |

#### Open questions to settle in the pilot (milestone M3)

1. **Billing of shared state.** Confirm from `usage.input_tokens` that shared `state` is billed once per request and not once per question. If it is billed per question, batching saves payload size but not tokens, and K should shrink.
2. **Cross-candidate interference.** Every question sees all K candidates in `state`. Compare accuracy at K = 1 against K = 8 against K = 16 on the same borderline set. If accuracy drops with K, lower the default K.
3. **Context limit.** Find the largest K that fits the context window at `max_side = 1024`. If the limit binds, either lower `max_side` or enable the long-context header.

### Step 6: Routing and report

- **Copy, never move.** Inputs stay untouched. Name collisions in the output get a `_<shortHash>` suffix.
- The output directory must not exist, or the user must pass `--overwrite`, which clears only folders this tool created.
- **Single label:** an identity belongs to one reference, so each image lands in exactly one folder.
- `report.csv` has these columns: `file, sha256, s_ref_1..s_ref_M, band, jev_choice, jev_probabilities, jev_batch_id, final_folder, error`.
- The console summary shows counts per folder, the number of Jev requests, total input tokens and an estimated cost (about $0.042 per 1M input tokens; confirm on the pricing page).

## 5. CLI

```
image-classifier classify INPUT_DIR --refs REF_DIR_OR_FILES... --out OUT_DIR
    [--config classifier.toml] [--model dinov2-base|siglip-base]
    [--t-high 0.70] [--t-low 0.45] [--margin 0.05]
    [--no-jev] [--jev-accept 0.80] [--jev-batch 16] [--max-jev-requests 50]
    [--dry-run]            # bands + estimated Jev requests, no copying, no API calls
    [--overwrite]

image-classifier calibrate LABELED_DIR --refs ...   # sweep thresholds, print precision/recall per band
```

`--dry-run` is the default first step on a new dataset. It shows how many images would go to Jev before spending anything.

## 6. Calibration

The thresholds are the product: wrong values either flood Jev or misfile images.

1. Build a labeled set of about 100–300 candidate images per domain, each labeled with its true reference (or none). Include hard negatives: same type, different individual.
2. `calibrate.py` computes the similarity matrix from the cache and sweeps `T_high`, `T_low` and `margin`. It reports:
   - precision of ACCEPT (target ≥ 0.98), the misfiling rate;
   - recall loss in REJECT (true matches thrown into `_unmatched`);
   - the BORDERLINE fraction, which is the Jev cost driver.
3. Run Jev on the borderline slice and sweep `J_accept` in the same way.
4. Choose between the CLS token and CLS + patch-mean, and between DINOv2 and SigLIP, on the same labeled set.
5. Save the chosen values to `classifier.toml`. Recalibrate whenever the domain changes.

## 7. Testing

| Level | What |
|---|---|
| Unit | Banding rules at the boundaries; Jev payload builder (each asset referenced exactly once, references only in `state`, question paths valid); batch packer byte limits; preprocess outputs inside the 280–1568 px and pixel-count bounds; collision-safe copy |
| Contract | Jev client against recorded fixtures: success, 422 bisect, 429 retry, and malformed or missing answers |
| Integration | A small fixture set (2 references, about 20 candidates including near-duplicates, same-type-different-identity and unrelated images) run end-to-end with Jev mocked |
| Live (opt-in) | `pytest -m live` sends one real Jev batch and is skipped without `TYPESAFE_API_KEY` |

## 8. Milestones

| # | Deliverable | Exit criterion |
|---|---|---|
| M1 | Scan, preprocess, DINOv2 embed + cache, banding, copy, report, `--no-jev` | Fixture set routes correctly; borderline → `_review/` |
| M2 | Calibration script + labeled set | Thresholds chosen; ACCEPT precision ≥ 0.98 on the labeled set |
| M3 | Jev judge: payload builder, batch packer, retries/bisect; **pilot** | Open questions in §4 step 5 answered; K and `J_accept` set from data |
| M4 | Hardening | Dry-run cost estimate, request cap, live test, README |

## 9. Risks

| Risk | Mitigation |
|---|---|
| Jev image support is experimental and its API may change | Jev isolated in `jev_judge.py`; `--no-jev` fallback; contract tests catch drift |
| Hard negatives (identical-looking products, twins, same model of car) | Both stages may fail; route uncertain cases to `_review/` rather than guessing. Consider more reference photos per identity (future: several images per reference folder, averaged). |
| Batch interference lowers Jev accuracy | Measured in M3; K tunable down to 1 |
| One malformed image fails a whole batch | Pre-validation plus bisect on 4xx |
| Only one reference photo per identity, with a pose very different from candidates | Flip augmentation now; multiple reference images per identity as a later extension |
