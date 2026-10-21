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

`classify` calls Jev only for BORDERLINE images. To send one candidate and the
references directly, skipping embeddings and banding:

```bash
export TYPESAFE_API_KEY=...
image-classifier jev photos/file.jpg --refs refs/ --config classifier.toml
```

On Windows, quote a path that contains spaces and leave off the trailing
backslash. A trailing `\` escapes the closing quote, so the next argument is
swallowed into the path:

```powershell
image-classifier jev 'C:\Users\daniel.hwang\Work\image-classifier\photos\file.jpg' --refs refs --config .\classifier.toml
```

This makes one real Jev request and copies nothing. It prints the model id,
request count, input tokens, Jev's `Choice`, and a `Decision`. `Decision` is
that choice when its probability is at least `jev.accept` (default 0.80);
otherwise it is `review`. Optional flags: `--jev-model`, `--jev-accept`.

`--refs` takes a directory or files and is repeatable (`--refs a.jpg --refs b.jpg`),
1–5 images in total. Folder names come from the reference filename stems. Rerunning
into the same output needs `--overwrite`, which deletes only the folders a
previous run created (tracked in `.image-classifier.json`).

Other options: `--config classifier.toml` (see `classifier.example.toml`), `--model
siglip-base`, `--patch-mean`, `--t-high/--t-low/--margin`, `--jev-accept`,
`--jev-batch`, `--jev-model`, `--no-recursive`, `-v`.

On Windows, write paths in `classifier.toml` with forward slashes or single quotes
(`cache_dir = 'C:\cache'`). In a double-quoted TOML string, `\U`, `\t` and so on
are escape sequences, so `"C:\Users\..."` fails to parse.

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

- **The Jev API contract is verified by a live run** (`pytest -m live`, 2026-10): the
  endpoint `/v1/systemone`, `Authorization: Bearer` auth, the request shape, and the
  response shape `answers.<qid>.{choice, probabilities}` plus `usage.input_tokens`.
  One request with 1 reference and 2 candidates used 1,465 input tokens.
- **Model id:** only `jev-latest` was accepted; `jev-1.13` and `jev-1.14` return
  `400 Unknown model`. `latest` is a moving alias, so a TypeSafe release can change
  answers and invalidate a calibrated `jev.accept`. Pin a versioned id when you have
  one (`--jev-model`, `[jev] model`, or `TYPESAFE_MODEL` for the live test).
- Image input is still experimental on TypeSafe's side; rerun `pytest -m live` after
  any upgrade or when Jev results look off.
- On a 400/422 the judge drops the candidates named in the error and resends the rest.
  It bisects only when the error is about an image but names none. Any other rejection
  (unknown model, bad field) disables Jev for the run, so a config error costs one
  request instead of one per bisection step.
- Payloads are base64 JSON. The plan's preferred multipart upload is not implemented,
  because its wire format is undocumented here. Expect bodies about 33% larger than
  the raw JPEGs.
- `jev.accept` (`J_accept`) and the batch size K still have to be set from the M3
  pilot. The calibrate command sweeps only the embedding thresholds.
- Calibration on real data has not been run; every shipped threshold is a placeholder.
