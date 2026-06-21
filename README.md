# Automated Image Variant Generator

> Built as a hands-on project to learn FastAPI, computer vision (DINOv2), and the full Git/GitHub workflow.
Upload a "master" marketing/product image and automatically generate
on-brand variants — different aspect ratios, brand-palette colour tints,
and background swaps — filtered for visual consistency using
[DINOv2](https://github.com/facebookresearch/dinov2) embeddings.

## How it works

Master image upload
→ Variant generator (aspect-ratio crop / palette tint / background swap)
→ DINOv2 embedding extraction (master + every variant)
→ Cosine similarity filter (flags variants that drifted too far from the master)
→ Storage (SQLite metadata + local file storage)
→ FastAPI backend → Web UI (gallery, similarity scores, ZIP/JSON export)

## Tech stack

- **Backend:** Python 3, FastAPI
- **AI model:** Meta's DINOv2 (`dinov2_vits14`), loaded via `torch.hub`
- **Image processing:** Pillow, OpenCV (GrabCut for background segmentation)
- **Storage:** SQLite + local filesystem
- **Frontend:** Vanilla HTML/CSS/JS (no framework)

## Features

- Upload any product/marketing image as a "master"
- Generate multiple aspect-ratio crops (1:1, 4:3, 16:9, 9:16, 3:4)
- Apply brand-colour palette tints
- Swap backgrounds behind the detected subject
- Every variant scored via DINOv2 cosine similarity against the master
- Variants below a configurable similarity threshold are auto-flagged for review
- Export per-master metadata as JSON
- Download master + all variants as a single ZIP

## Setup

```bash
git clone https://github.com/Vijay-ship-it-cloud/Image_variant_generator.git
cd Image_variant_generator
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The first run downloads the DINOv2 model weights automatically (one-time, ~90MB).

## Run

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

## Project structure