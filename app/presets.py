from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .crop import Crop
from .detector import FaceBox


@dataclass(frozen=True, slots=True)
class Preset:
    """Jeu de valeurs par défaut pour `w`, `h` et `zoom`."""

    w: int
    h: int
    zoom: float
    description: str


# 413x531 = 35x45 mm à 300 dpi, le format des photos d'identité ICAO 9303 / ANTS.
# zoom 1.9 place la tête (crâne→menton) à ~76 % de la hauteur du cadre, dans la
# fourchette 70-80 % exigée par la norme.
PRESETS: dict[str, Preset] = {
    "id": Preset(
        w=413,
        h=531,
        zoom=1.9,
        description="Photo d'identité 35x45 mm à 300 dpi, tête à ~76 % du cadre",
    ),
}

# Rapport hauteur de tête (crâne→menton) sur hauteur de la boîte YuNet, qui
# cadre approximativement des sourcils au menton.
HEAD_TO_BOX = 1.45

# En deçà, la détection est trop incertaine pour un usage identité.
MIN_SCORE = 0.90


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
) -> list[str]:
    """Contrôles indicatifs de conformité pour le preset `id`.

    Ne portent que sur ce qui est mesurable à partir de la détection et du
    cadrage : nombre de visages, confiance, résolution, recadrage tronqué. La
    norme exige aussi un fond uni, une expression neutre et un regard vers
    l'objectif, qu'aucun de ces contrôles ne vérifie.
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

    # `plan` vise une hauteur de `zoom * hauteur de visage` ; si le résultat est
    # plus court, c'est que le cadre a été tronqué par les bords de l'image.
    if crop.h < zoom * subject.h * 0.98:
        head_ratio = subject.h * HEAD_TO_BOX / crop.h * 100
        warnings.append(
            f"cadre tronqué par les bords de l'image : tête à {head_ratio:.0f}% "
            "du cadre au lieu de 70-80%"
        )

    return warnings
