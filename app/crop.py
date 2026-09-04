from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .detector import FaceBox

# Décalage vertical du centre du crop, en fraction de la hauteur du visage.
# Positif = on remonte le cadre pour laisser de l'air au-dessus de la tête.
HEAD_ROOM = 0.15

# Le visage doit tenir confortablement en largeur.
MIN_WIDTH_FACTOR = 1.25


@dataclass(frozen=True, slots=True)
class Crop:
    """Rectangle de crop dans le repère de l'image source."""

    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    def to_openinary(self) -> str:
        """Segment de transformation Openinary, ex. `c_crop,x_312,y_98,w_540,h_540`."""
        return f"c_crop,x_{self.x},y_{self.y},w_{self.w},h_{self.h}"


@dataclass(frozen=True, slots=True)
class _Box:
    x: float
    y: float
    w: float
    h: float


def _union(faces: Sequence[FaceBox]) -> _Box:
    """Boîte englobant tous les visages."""
    x0 = min(f.x for f in faces)
    y0 = min(f.y for f in faces)
    x1 = max(f.x + f.w for f in faces)
    y1 = max(f.y + f.h for f in faces)
    return _Box(x=float(x0), y=float(y0), w=float(x1 - x0), h=float(y1 - y0))


def _centered(img_w: int, img_h: int, ratio: float) -> Crop:
    """Crop centré le plus grand possible au ratio demandé (fallback sans visage)."""
    cw = float(img_w)
    ch = cw / ratio
    if ch > img_h:
        ch = float(img_h)
        cw = ch * ratio
    return _finalize(img_w, img_h, img_w / 2.0, img_h / 2.0, cw, ch, ratio)


def _finalize(
    img_w: int,
    img_h: int,
    cx: float,
    cy: float,
    cw: float,
    ch: float,
    ratio: float,
) -> Crop:
    """Arrondit, contraint au ratio et clampe le rectangle dans l'image."""
    # Ne jamais dépasser l'image, tout en conservant le ratio.
    if cw > img_w:
        cw = float(img_w)
        ch = cw / ratio
    if ch > img_h:
        ch = float(img_h)
        cw = ch * ratio

    w = max(1, min(img_w, int(math.floor(cw))))
    h = max(1, min(img_h, int(math.floor(round(w / ratio, 6)))))
    if h > img_h:
        h = img_h
        w = max(1, min(img_w, int(math.floor(h * ratio))))

    x = int(round(cx - w / 2.0))
    y = int(round(cy - h / 2.0))
    x = max(0, min(img_w - w, x))
    y = max(0, min(img_h - h, y))
    return Crop(x=x, y=y, w=w, h=h)


def plan(
    img_w: int,
    img_h: int,
    faces: Sequence[FaceBox],
    target_w: int,
    target_h: int,
    zoom: float = 2.6,
    all_faces: bool = False,
) -> Crop:
    """Calcule le rectangle de crop centré visage, façon `g_face,c_fill` de Cloudinary.

    `zoom` exprime la hauteur du crop en multiples de la hauteur du visage
    (2.6 ≈ portrait serré, 4+ ≈ buste). Sans visage détecté, on retombe sur un
    crop centré au bon ratio — jamais d'exception.
    """
    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))
    ratio = max(target_w, 1) / max(target_h, 1)

    if not faces:
        return _centered(img_w, img_h, ratio)

    # Sans `all_faces`, le sujet principal est le plus grand visage (le détecteur
    # trie déjà ainsi, mais `plan` ne dépend pas de l'ordre reçu).
    box = _union(faces) if all_faces else _union([max(faces, key=lambda f: f.area)])

    ch = max(1.0, float(zoom)) * box.h
    cw = ch * ratio

    # Garantir que le visage tient en largeur.
    min_w = MIN_WIDTH_FACTOR * box.w
    if cw < min_w:
        cw = min_w
        ch = cw / ratio

    cx = box.x + box.w / 2.0
    cy = box.y + box.h / 2.0 - HEAD_ROOM * box.h

    return _finalize(img_w, img_h, cx, cy, cw, ch, ratio)
