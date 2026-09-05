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

# Anthropométrie, exprimée en distances yeux→bouche (`d`). Les yeux tombent à peu
# près au milieu vertical de la tête : le crâne est ~1.70 d au-dessus, le menton
# ~1.60 d en dessous, soit une hauteur de tête de ~3.30 d. Ancré sur des points de
# repère, c'est nettement plus stable que la hauteur de boîte YuNet, qui varie avec
# la pose et l'éclairage.
CROWN_ABOVE_EYES = 1.70
CHIN_BELOW_EYES = 1.60
HEAD_FROM_EYE_MOUTH = CROWN_ABOVE_EYES + CHIN_BELOW_EYES


@dataclass(frozen=True, slots=True)
class HeadMetrics:
    """Géométrie de la tête déduite des points de repère, en pixels source."""

    eye_x: float
    eye_y: float
    crown_y: float
    chin_y: float
    roll_degrees: float

    @property
    def head_height(self) -> float:
        return self.chin_y - self.crown_y


def measure(face: FaceBox) -> HeadMetrics | None:
    """Mesure la tête à partir des landmarks, ou None si le détecteur n'en a pas."""
    lm = face.landmarks
    if lm is None:
        return None
    d = lm.eye_mouth_distance
    if d <= 0:
        return None
    eye_x, eye_y = lm.eye_center
    return HeadMetrics(
        eye_x=eye_x,
        eye_y=eye_y,
        crown_y=eye_y - CROWN_ABOVE_EYES * d,
        chin_y=eye_y + CHIN_BELOW_EYES * d,
        roll_degrees=lm.roll_degrees,
    )


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


def plan_by_head(
    img_w: int,
    img_h: int,
    metrics: HeadMetrics,
    target_w: int,
    target_h: int,
    head_ratio: float,
    eye_line: float,
) -> Crop:
    """Cadre sur la tête mesurée, façon norme photo d'identité.

    `head_ratio` est la fraction de la hauteur du cadre qu'occupe la tête
    (crâne→menton) ; `eye_line` la hauteur de la ligne des yeux mesurée depuis le
    bas du cadre. Contrairement à `plan`, rien n'est déduit de la boîte englobante.
    """
    img_w = max(1, int(img_w))
    img_h = max(1, int(img_h))
    ratio = max(target_w, 1) / max(target_h, 1)

    ch = metrics.head_height / max(head_ratio, 0.01)
    cw = ch * ratio

    # Le centre vertical se déduit de la position voulue pour la ligne des yeux.
    cy = metrics.eye_y + ch * (eye_line - 0.5)
    return _finalize(img_w, img_h, metrics.eye_x, cy, cw, ch, ratio)


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
