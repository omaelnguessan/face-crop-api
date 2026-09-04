#!/usr/bin/env bash
# Télécharge le modèle YuNet (~230 Ko) depuis opencv/opencv_zoo vers models/.
set -euo pipefail

MODEL_NAME="face_detection_yunet_2023mar.onnx"
# Le fichier est stocké en Git LFS : media.githubusercontent.com sert le binaire réel,
# raw.githubusercontent.com ne renverrait qu'un pointeur LFS de ~130 octets.
BASE_URL="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${ROOT_DIR}/models"
DEST="${DEST_DIR}/${MODEL_NAME}"

mkdir -p "${DEST_DIR}"

if [[ -f "${DEST}" ]]; then
  echo "✓ Modèle déjà présent : ${DEST}"
  exit 0
fi

echo "→ Téléchargement de ${MODEL_NAME}…"
if ! curl -fL --retry 3 --connect-timeout 10 -o "${DEST}.tmp" "${BASE_URL}/${MODEL_NAME}"; then
  rm -f "${DEST}.tmp"
  cat >&2 <<'MSG'
✗ Échec du téléchargement du modèle YuNet.
  Vérifie ta connexion, puis récupère-le manuellement depuis :
  https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
  et place face_detection_yunet_2023mar.onnx dans models/.
MSG
  exit 1
fi

# Garde-fou : un pointeur Git LFS fait ~130 octets, le vrai modèle ~230 Ko.
SIZE=$(wc -c < "${DEST}.tmp" | tr -d ' ')
if [[ "${SIZE}" -lt 100000 ]]; then
  rm -f "${DEST}.tmp"
  echo "✗ Fichier téléchargé invalide (${SIZE} octets, attendu ~230 Ko)." >&2
  echo "  Récupère-le manuellement : ${BASE_URL}/${MODEL_NAME}" >&2
  exit 1
fi

mv "${DEST}.tmp" "${DEST}"
echo "✓ Modèle installé : ${DEST} ($(du -h "${DEST}" | cut -f1))"
