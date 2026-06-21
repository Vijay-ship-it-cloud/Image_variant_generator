"""
main.py — FastAPI backend.
Run with: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.modules import storage
from app.modules.variant_generator import BRAND_PALETTES
from app.modules.variant_pipeline import default_variant_specs, generate_variants_for_master

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Automated Image Variant Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    storage.init_db()


@app.get("/")
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/api/palettes")
def get_palettes():
    return {"palettes": list(BRAND_PALETTES.keys())}


@app.post("/api/masters")
async def upload_master(file: UploadFile = File(...), brand: str | None = Form(None)):
    image_bytes = await file.read()

    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    max_dim = 2048
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_bytes = buf.getvalue()

    record = storage.save_master(filename=file.filename, image_bytes=image_bytes, brand=brand)
    return record


@app.get("/api/masters")
def list_masters():
    return {"masters": storage.list_masters()}


@app.post("/api/masters/{master_id}/generate")
def generate_variants(
    master_id: str,
    aspect_ratios: str = Form("1:1,4:3,16:9"),
    palettes: str = Form(""),
    similarity_threshold: float = Form(0.90),
    background_color: str = Form(""),
):
    master = storage.get_master(master_id)
    if not master:
        raise HTTPException(status_code=404, detail="Master image not found.")

    master_image = Image.open(master["filepath"]).convert("RGB")

    ratio_list = [r.strip() for r in aspect_ratios.split(",") if r.strip()]
    palette_list = [p.strip() for p in palettes.split(",") if p.strip()]

    bg_color = None
    if background_color:
        try:
            r, g, b = (int(x) for x in background_color.split(","))
            bg_color = (r, g, b)
        except ValueError:
            raise HTTPException(status_code=400, detail="background_color must be 'r,g,b'")

    specs = default_variant_specs(
        aspect_ratios=ratio_list,
        palettes=palette_list,
        include_background=bg_color is not None,
        background_color=bg_color,
    )

    results, summary = generate_variants_for_master(
        master_id=master_id,
        master_image=master_image,
        specs=specs,
        similarity_threshold=similarity_threshold,
    )

    return {
        "summary": summary,
        "variants": [
            {
                "id": r.db_record["id"],
                "filename": r.db_record["filename"],
                "variant_type": r.variant_type,
                "aspect_ratio": r.aspect_ratio,
                "similarity_score": round(r.similarity_score, 4),
                "passed_filter": r.passed_filter,
                "url": f"/api/variants/{r.db_record['id']}/image",
            }
            for r in results
        ],
    }


@app.get("/api/masters/{master_id}/variants")
def list_variants(master_id: str):
    if not storage.get_master(master_id):
        raise HTTPException(status_code=404, detail="Master image not found.")
    variants = storage.list_variants_for_master(master_id)
    for v in variants:
        v["url"] = f"/api/variants/{v['id']}/image"
    return {"variants": variants}


@app.get("/api/masters/{master_id}/image")
def get_master_image(master_id: str):
    master = storage.get_master(master_id)
    if not master:
        raise HTTPException(status_code=404, detail="Master image not found.")
    return FileResponse(master["filepath"])


@app.get("/api/variants/{variant_id}/image")
def get_variant_image(variant_id: str):
    with storage.get_conn() as conn:
        row = conn.execute("SELECT * FROM variants WHERE id = ?", (variant_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Variant not found.")
    return FileResponse(row["filepath"])


@app.get("/api/masters/{master_id}/export")
def export_metadata(master_id: str):
    if not storage.get_master(master_id):
        raise HTTPException(status_code=404, detail="Master image not found.")
    payload = storage.export_metadata_json(master_id)
    return Response(content=payload, media_type="application/json")


@app.get("/api/masters/{master_id}/zip")
def download_zip(master_id: str):
    master = storage.get_master(master_id)
    if not master:
        raise HTTPException(status_code=404, detail="Master image not found.")

    variants = storage.list_variants_for_master(master_id)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(master["filepath"], arcname=f"master_{master['filename']}")
        for v in variants:
            zf.write(v["filepath"], arcname=v["filename"])
        zf.writestr("metadata.json", storage.export_metadata_json(master_id))

    buf.seek(0)
    return Response(
        content=buf.read(),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=variants_{master_id}.zip"},
    )


@app.get("/api/logs")
def get_logs():
    return {"logs": storage.get_logs()}