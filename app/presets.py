from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .crop import Crop, HeadMetrics
from .detector import FaceBox


@dataclass(frozen=True, slots=True)
class Preset:
    """Jeu de valeurs par défaut pour `w`, `h` et `zoom`.

    `head_ratio` et `eye_line` ne servent que si le détecteur a fourni des points
    de repère : le cadrage est alors mesuré (`plan_by_head`) plutôt que déduit du
    `zoom`, qui reste le repli.
    """

    w: int
    h: int
    zoom: float
    description: str
    head_ratio: float | None = None
    eye_line: float | None = None


# 413x531 = 35x45 mm à 300 dpi, le format des photos d'identité ICAO 9303 / ANTS.
# zoom 1.9 place la tête (crâne→menton) à ~76 % de la hauteur du cadre, dans la
# fourchette 70-80 % exigée par la norme.
PRESETS: dict[str, Preset] = {
    "id": Preset(
        w=413,
        h=531,
        zoom=1.9,
        description="Photo d'identité 35x45 mm à 300 dpi, tête à ~76 % du cadre",
        # Milieux des fourchettes ICAO : tête 70-80 %, ligne des yeux 50-65 %.
        head_ratio=0.76,
        eye_line=0.575,
    ),
}

# En deçà, la détection est trop incertaine pour un usage identité.
MIN_SCORE = 0.90

# Fourchettes ICAO 9303 vérifiées quand la tête a pu être mesurée.
HEAD_RATIO_RANGE = (0.70, 0.80)
EYE_LINE_RANGE = (0.50, 0.65)

# Au-delà, la tête n'est plus « droite » au sens de la norme.
MAX_ROLL_DEGREES = 5.0


def resolve(name: str | None) -> Preset | None:
    """Retourne le preset demandé, ou None si aucun n'est demandé.

    Lève `KeyError` sur un nom inconnu — l'appelant le traduit en HTTP 400.
    """
    if name is None:
        return None
    return PRESETS[name.strip().lower()]


def id_warnings(
    faces: Sequence[FaceBox],
    crop: Crop,
    target_w: int,
    target_h: int,
    zoom: float,
    metrics: HeadMetrics | None = None,
) -> list[str]:
    """Contrôles indicatifs de conformité pour le preset `id`.

    Avec `metrics` (points de repère disponibles), le cadrage est vérifié sur des
    valeurs mesurées : hauteur de tête réelle, ligne des yeux, inclinaison. Sans,
    on se limite à ce que la boîte englobante permet de dire.

    Ces contrôles ne portent que sur la géométrie et la qualité de détection. La
    norme exige aussi un fond uni, une expression neutre et un regard vers
    l'objectif, qu'aucun d'eux ne vérifie.
    """
    warnings: list[str] = []

    if not faces:
        warnings.append(
            "aucun visage détecté : cadrage centré par défaut, non conforme"
        )
        return warnings

    if len(faces) > 1:
        warnings.append(
            f"{len(faces)} visages détectés : une photo d'identité ne doit en contenir qu'un"
        )

    subject = max(faces, key=lambda f: f.area)

    if subject.score < MIN_SCORE:
        warnings.append(
            f"confiance de détection faible ({subject.score:.2f} < {MIN_SCORE:.2f})"
        )

    if crop.w < target_w or crop.h < target_h:
        warnings.append(
            f"résolution source insuffisante : crop {crop.w}x{crop.h} "
            f"agrandi vers {target_w}x{target_h}"
        )

    if metrics is None:
        # Repli sans points de repère : `plan` vise `zoom * hauteur de visage`,
        # un résultat plus court signale un cadre tronqué par les bords.
        if crop.h < zoom * subject.h * 0.98:
            warnings.append(
                "cadre tronqué par les bords de l'image : cadrage non conforme"
            )
        return warnings

    head_ratio = metrics.head_height / crop.h
    lo, hi = HEAD_RATIO_RANGE
    if not (lo <= head_ratio <= hi):
        warnings.append(
            f"tête à {head_ratio * 100:.0f}% du cadre, hors norme "
            f"({lo * 100:.0f}-{hi * 100:.0f}%) : image trop courte pour le cadrage visé"
        )

    eye_line = (crop.y + crop.h - metrics.eye_y) / crop.h
    lo, hi = EYE_LINE_RANGE
    if not (lo <= eye_line <= hi):
        warnings.append(
            f"ligne des yeux à {eye_line * 100:.0f}% du bas, hors norme "
            f"({lo * 100:.0f}-{hi * 100:.0f}%)"
        )

    roll = abs(metrics.roll_degrees)
    if roll > MAX_ROLL_DEGREES:
        warnings.append(
            f"tête inclinée de {roll:.1f}°, au-delà de {MAX_ROLL_DEGREES:.0f}° tolérés"
        )

    return warnings
