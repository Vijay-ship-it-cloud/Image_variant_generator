"""
variant_pipeline.py
--------------------
Orchestrates: generate variants -> embed each via DINOv2 -> cosine similarity
vs master -> filter/score -> store -> log.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

from PIL import Image

from . import storage
from .embeddings import cosine_similarity, get_extractor
from .variant_generator import VariantSpec, apply_variant_spec

DEFAULT_SIMILARITY_THRESHOLD = 0.90


@dataclass
class VariantResult:
    label: str
    variant_type: str
    aspect_ratio: str | None
    similarity_score: float
    passed_filter: bool
    image_bytes: bytes
    db_record: dict


def default_variant_specs(
    aspect_ratios: list[str] | None = None,
    palettes: list[str] | None = None,
    include_background: bool = False,
    background_color: tuple[int, int, int] | None = None,
) -> list[VariantSpec]:
    specs: list[VariantSpec] = []

    for ratio in aspect_ratios or ["1:1", "4:3", "16:9"]:
        specs.append(VariantSpec(kind="aspect_ratio", params={"ratio_name": ratio}))

    for palette in palettes or []:
        specs.append(VariantSpec(kind="palette", params={"palette_name": palette, "intensity": 0.25}))

    if include_background and background_color is not None:
        specs.append(VariantSpec(kind="background", params={"color": background_color, "label": "flat"}))

    return specs


def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format=fmt)
    return buf.getvalue()


def generate_variants_for_master(
    master_id: str,
    master_image: Image.Image,
    specs: list[VariantSpec],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    max_variants: int = 10,
) -> tuple[list[VariantResult], dict]:
    start = time.time()
    extractor = get_extractor()

    specs = specs[:max_variants]
    master_embedding = extractor.embed_image(master_image)

    results: list[VariantResult] = []
    filtered_out_count = 0

    for idx, spec in enumerate(specs):
        try:
            variant_image = apply_variant_spec(master_image, spec)
        except Exception:
            continue

        variant_embedding = extractor.embed_image(variant_image)
        similarity = cosine_similarity(master_embedding, variant_embedding)
        passed = similarity >= similarity_threshold
        if not passed:
            filtered_out_count += 1

        image_bytes = _image_to_bytes(variant_image)
        filename = f"{idx:02d}_{spec.label}.png"
        aspect_ratio = spec.params.get("ratio_name") if spec.kind == "aspect_ratio" else None

        db_record = storage.save_variant(
            master_id=master_id,
            filename=filename,
            image_bytes=image_bytes,
            variant_type=spec.kind,
            similarity_score=similarity,
            passed_filter=passed,
            aspect_ratio=aspect_ratio,
        )

        results.append(
            VariantResult(
                label=spec.label,
                variant_type=spec.kind,
                aspect_ratio=aspect_ratio,
                similarity_score=similarity,
                passed_filter=passed,
                image_bytes=image_bytes,
                db_record=db_record,
            )
        )

    elapsed = time.time() - start
    storage.log_run(
        master_id=master_id,
        variants_generated=len(results),
        variants_filtered_out=filtered_out_count,
        processing_time_sec=elapsed,
    )

    summary = {
        "variants_generated": len(results),
        "variants_filtered_out": filtered_out_count,
        "processing_time_sec": elapsed,
    }
    return results, summary