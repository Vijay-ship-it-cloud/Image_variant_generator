"""
variant_generator.py
---------------------
Pure image-processing module: takes a master PIL image and produces variant
candidates. No model inference happens here.

Three variant families:
1. Aspect-ratio crops    (1:1, 4:3, 16:9, 9:16, 3:4)
2. Colour/palette shifts (brand-colour tint or hue rotation)
3. Background swap       (GrabCut foreground cutout + new background)
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageEnhance

# ---------------------------------------------------------------------------
# Aspect ratio crops
# ---------------------------------------------------------------------------

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "3:4": (3, 4),
}


def crop_to_aspect(image: Image.Image, ratio_name: str) -> Image.Image:
    """Centre-crops `image` to the requested aspect ratio (no stretching)."""
    if ratio_name not in ASPECT_RATIOS:
        raise ValueError(f"Unknown aspect ratio '{ratio_name}'. Choose from {list(ASPECT_RATIOS)}")

    target_w_ratio, target_h_ratio = ASPECT_RATIOS[ratio_name]
    target_ratio = target_w_ratio / target_h_ratio

    w, h = image.size
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        box = (left, 0, left + new_w, h)
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        box = (0, top, w, top + new_h)

    return image.crop(box)


# ---------------------------------------------------------------------------
# Colour / brand palette shifts
# ---------------------------------------------------------------------------

BRAND_PALETTES: dict[str, tuple[int, int, int]] = {
    "ocean_blue": (24, 98, 168),
    "sunset_orange": (230, 126, 34),
    "forest_green": (39, 119, 75),
    "royal_purple": (108, 52, 161),
    "crimson_red": (192, 41, 66),
    "slate_gray": (90, 99, 110),
    "golden_yellow": (212, 160, 23),
}


def apply_palette_tint(image: Image.Image, palette_name: str, intensity: float = 0.25) -> Image.Image:
    """Tints the image toward a brand colour. Intensity is capped at 0.6
    so the transformation stays subtle and brand identity isn't lost."""
    if palette_name not in BRAND_PALETTES:
        raise ValueError(f"Unknown palette '{palette_name}'. Choose from {list(BRAND_PALETTES)}")

    intensity = max(0.0, min(intensity, 0.6))
    rgb = BRAND_PALETTES[palette_name]
    image = image.convert("RGB")
    tint_layer = Image.new("RGB", image.size, rgb)
    return Image.blend(image, tint_layer, intensity)


def apply_hue_shift(image: Image.Image, degrees: float) -> Image.Image:
    """Rotates the hue of the whole image by `degrees` (-180..180)."""
    image = image.convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*pixel) for pixel in arr.reshape(-1, 3)])
    hsv[:, 0] = (hsv[:, 0] + degrees / 360.0) % 1.0
    rgb = np.array([colorsys.hsv_to_rgb(*pixel) for pixel in hsv])
    rgb = (rgb.reshape(arr.shape) * 255).astype(np.uint8)
    return Image.fromarray(rgb)


def adjust_lighting(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0) -> Image.Image:
    """Minor lighting/style adjustment."""
    image = ImageEnhance.Brightness(image).enhance(brightness)
    image = ImageEnhance.Contrast(image).enhance(contrast)
    return image


# ---------------------------------------------------------------------------
# Background swap
# ---------------------------------------------------------------------------

def _estimate_foreground_mask(image: Image.Image) -> np.ndarray:
    """Foreground/background separation via OpenCV GrabCut, seeded with a
    centred rectangle. Works well for centred product/marketing shots.
    Returns a single-channel mask (0..255) where 255 = foreground."""
    cv_img = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = cv_img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    rect = (margin_w, margin_h, w - 2 * margin_w, h - 2 * margin_h)

    try:
        cv2.grabCut(cv_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.full((h, w), 255, dtype=np.uint8)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
    return fg_mask


def swap_background(image: Image.Image, background: Image.Image | tuple[int, int, int]) -> Image.Image:
    """Replaces the background behind the detected foreground subject.
    `background` can be a flat RGB colour or another PIL image."""
    image = image.convert("RGB")
    w, h = image.size
    mask = _estimate_foreground_mask(image)

    if isinstance(background, tuple):
        bg_img = Image.new("RGB", (w, h), background)
    else:
        bg_img = background.convert("RGB").resize((w, h))

    mask_img = Image.fromarray(mask).convert("L")
    return Image.composite(image, bg_img, mask_img)


# ---------------------------------------------------------------------------
# Variant spec / batch description
# ---------------------------------------------------------------------------

@dataclass
class VariantSpec:
    """kind is one of: aspect_ratio, palette, hue, background."""
    kind: str
    params: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        if self.kind == "aspect_ratio":
            return f"aspect_{self.params['ratio_name'].replace(':', 'x')}"
        if self.kind == "palette":
            return f"palette_{self.params['palette_name']}"
        if self.kind == "hue":
            return f"hue_{int(self.params['degrees'])}"
        if self.kind == "background":
            return f"bg_{self.params.get('label', 'swap')}"
        return self.kind


def apply_variant_spec(image: Image.Image, spec: VariantSpec) -> Image.Image:
    if spec.kind == "aspect_ratio":
        return crop_to_aspect(image, spec.params["ratio_name"])
    if spec.kind == "palette":
        return apply_palette_tint(image, spec.params["palette_name"], spec.params.get("intensity", 0.25))
    if spec.kind == "hue":
        return apply_hue_shift(image, spec.params["degrees"])
    if spec.kind == "background":
        bg = spec.params.get("color") or spec.params.get("image")
        return swap_background(image, bg)
    raise ValueError(f"Unknown variant kind: {spec.kind}")