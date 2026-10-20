# image-classifier

Sorts a directory of images into one folder per reference image by **identity**:
same individual object or subject, not merely the same kind of thing. See
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the design.

1. **DINOv2 embeddings (local, every image).** Cosine similarity to each reference
   bands every candidate as ACCEPT, REJECT or BORDERLINE.
2. **TypeSafe Jev (borderline only, optional).** Borderline candidates are packed
   into requests of up to 16 questions. The references are sent once per request
   in shared `state`.

Output: `OUT/<ref_name>/`, `OUT/_unmatched/`, `OUT/_review/` and `OUT/report.csv`.
Inputs are copied, never moved.

## Install

```bash
pip install -e '.[dev]'          # add ,heic for .heic input
```

The DINOv2 weights (~350 MB) download from Hugging Face on first run.

## Use

```bash
# 1. Always dry-run first: bands + how many Jev requests it would cost.
image-classifier classify photos/ --refs refs/ --out sorted/ --dry-run

# 2. Local only: borderline images go to _review/.
image-classifier classify photos/ --refs refs/ --out sorted/ --no-jev

# 3. With Jev on the borderline band.
export TYPESAFE_API_KEY=...
image-classifier classify photos/ --refs refs/ --out sorted/ --max-jev-requests 20
```

`--refs` takes a directory or files and is repeatable (`--refs a.jpg --refs b.jpg`),
1–5 images in total. Folder names come from the reference filename stems. Rerunning
into the same output needs `--overwrite`, which deletes only the folders a
previous run created (tracked in `.image-classifier.json`).

Other options: `--config classifier.toml` (see `classifier.example.toml`), `--model
siglip-base`, `--patch-mean`, `--t-high/--t-low/--margin`, `--jev-accept`,
`--jev-batch`, `--no-recursive`, `-v`.

Embeddings are cached in `.cache/embeddings/<model>/<sha256>.npy`, so changing
thresholds and rerunning costs no inference.

## Calibrate (do this before trusting any threshold)

```
labeled/
  red_bike/   *.jpg   # true matches of reference red_bike
  blue_bike/  *.jpg
  none/       *.jpg   # non-matches; include same-type-different-individual
```

```bash
image-classifier calibrate labeled/ --refs refs/     # or: python calibrate.py labeled/ --refs refs/
```

This prints the current config's ACCEPT precision, REJECT recall loss and borderline
fraction. It then lists the cheapest threshold sets that reach precision ≥ 0.98 with
recall loss ≤ 0.02 (`--min-precision` / `--max-recall-loss`) and gives a `[band]`
block to paste into `classifier.toml`. Rerun it with `--model siglip-base` and/or
`--patch-mean` to compare feature variants.

## Test

```bash
pytest                      # unit + contract + integration, Jev mocked, no weights needed
TYPESAFE_API_KEY=... pytest -m live   # one real Jev batch
```

## Known limitations / unverified assumptions

- **The Jev API contract is assumed, not verified.** The endpoint (`/v1/systemone`),
  the `Authorization: Bearer` auth, the request shape and the response shape
  (`answers.<qid>.{choice, confidence, probabilities}`, `usage.input_tokens`) follow
  the plan. They have not been checked against live docs, and all of them live in
  `image_classifier/jev_judge.py`. Run `pytest -m live` before relying on Jev.
- Payloads are base64 JSON. The plan's preferred multipart upload is not implemented,
  because its wire format is undocumented here. Expect bodies about 33% larger than
  the raw JPEGs.
- `jev.accept` (`J_accept`) and the batch size K still have to be set from the M3
  pilot. The calibrate command sweeps only the embedding thresholds.
- Calibration on real data has not been run; every shipped threshold is a placeholder.
