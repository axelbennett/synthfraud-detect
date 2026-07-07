# SynthFraud Detect

Explainable, defense-in-depth system for detecting AI-generated or manipulated
images submitted in auto insurance damage claims.

## Problem

Generative AI now makes it possible to fabricate convincing car damage photos,
inflating or entirely faking insurance claims. Insurance fraud costs U.S.
consumers an estimated $308.6B annually, and industry surveys show most
fraud teams don't yet feel prepared for AI-generated fraud. A single
"AI-or-not" classifier is not a durable solution — new generators constantly
outpace detectors. This project instead architects a multi-signal system
that combines weak signals, produces an explainable risk score, and routes
uncertain claims to a human reviewer rather than auto-approving or
auto-denying.

## Architecture

```
Claim photo + claim text
        |
        v
[1] Ingestion & preprocessing  -> normalize image, extract EXIF
        |
        v
[2] Forensic classifier (PyTorch CNN) -> P(AI-generated | image)
        |
        v
[3] Metadata consistency check -> EXIF / camera / timestamp anomalies
        |
        v
[4] Semantic consistency check (LLM) -> does damage match claim narrative?
        |
        v
[5] Risk scoring & fusion -> combined confidence score + explanation
        |
        v
[6] Review dashboard -> human-in-the-loop decision, never auto-deny
```

See `docs/architecture.md` (to be written) for full design rationale,
including why this is framed as fraud *triage*, not fraud *determination*.

## Project status

Early scaffold. Build order:

1. [ ] Pull CarDD (real damage photos) — `src/data/download_cardd.py`
2. [ ] Generate synthetic damage photos from 2+ generators — `src/data/generate_synthetic.py`
3. [ ] Train forensic classifier — `src/models/forensic_classifier.py`
4. [ ] Evaluate on held-out generator (generalization test) — `src/models/evaluate.py`
5. [ ] Semantic consistency check via LLM — `src/models/semantic_check.py`
6. [ ] Fusion + scoring API — `src/api/main.py`
7. [ ] Review dashboard (simple React or Streamlit)

## Repo structure

```
configs/            # config files (paths, hyperparams, generator settings)
data/
  raw/               # CarDD real images (gitignored, download separately)
  synthetic/         # generated fake images (gitignored)
src/
  data/              # dataset download + synthetic generation scripts
  models/            # forensic classifier, semantic check, fusion logic
  api/               # FastAPI service exposing the scoring pipeline
  utils/             # shared helpers (image I/O, logging, metrics)
tests/               # unit tests
notebooks/           # exploration notebooks
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt --break-system-packages
```

## Data sourcing

Real images: CarDD dataset (4,000 annotated car damage photos).
Mirror available on Hugging Face: `harpreetsahota/CarDD` (no license form needed).
Official source (requires license form): https://cardd-ustc.github.io/

Synthetic images: generated via Stable Diffusion / SDXL and a second
generator family (e.g. DALL-E), plus img2img edits of real CarDD photos to
simulate partial/touched-up fraud — the harder and more realistic case.

## Honest limitations (documented up front, not discovered later)

- No forensic classifier generalizes perfectly to generators unseen in
  training; this project measures and reports that degradation rather than
  hiding it.
- This is a proof-of-concept / portfolio project, not a production fraud
  system — no legal, compliance, or claims-payout logic is implemented.
