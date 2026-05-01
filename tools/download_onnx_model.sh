#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_url="${FOUNDATIONPOSE_ONNX_MODEL_URL:-https://drive.google.com/drive/folders/1AmBopDz-RrykSZVCroDH6jFc1-k8HkL0?usp=drive_link}"
models_dir="${FOUNDATIONPOSE_MODELS_DIR:-${repo_root}/models}"
download_dir="${FOUNDATIONPOSE_ONNX_DOWNLOAD_DIR:-${models_dir}/.download}"
scorer_onnx="${models_dir}/scorer_hwc.onnx"
refiner_onnx="${models_dir}/refiner_hwc.onnx"

extract_zip() {
  local archive="$1"
  local destination="$2"

  python - "$archive" "$destination" <<'PY'
import sys
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
destination.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(archive) as zf:
    zf.extractall(destination)
PY
}

extract_archive() {
  local archive="$1"
  local destination="$2"

  case "${archive}" in
    *.zip)
      extract_zip "${archive}" "${destination}"
      ;;
    *.tar|*.tar.gz|*.tgz|*.tar.bz2|*.tbz2|*.tar.xz|*.txz)
      tar -xf "${archive}" -C "${destination}"
      ;;
    *)
      return 1
      ;;
  esac
}

install_onnx_file() {
  local candidate="$1"
  local destination="${models_dir}/$(basename "${candidate}")"

  if [[ "${candidate}" == "${destination}" ]]; then
    return 0
  fi

  if [[ -f "${candidate}" && ! -f "${destination}" ]]; then
    mv "${candidate}" "${destination}"
  fi
}

if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing gdown. Run this through Pixi with: pixi run download-onnx-model" >&2
  exit 1
fi

if [[ -f "${scorer_onnx}" && -f "${refiner_onnx}" && "${FOUNDATIONPOSE_FORCE_DOWNLOAD:-0}" != "1" ]]; then
  echo "ONNX models already exist in ${models_dir}"
  echo "Set FOUNDATIONPOSE_FORCE_DOWNLOAD=1 to download again."
  exit 0
fi

mkdir -p "${models_dir}" "${download_dir}"

echo "Downloading FoundationPose ONNX models"
echo "  from: ${model_url}"
echo "  to:   ${models_dir}"

gdown --folder --continue -O "${download_dir}/" "${model_url}"

while IFS= read -r -d '' archive; do
  echo "Extracting ${archive}"
  extract_archive "${archive}" "${models_dir}"
done < <(
  find "${download_dir}" -type f \
    \( -name '*.zip' \
    -o -name '*.tar' \
    -o -name '*.tar.gz' \
    -o -name '*.tgz' \
    -o -name '*.tar.bz2' \
    -o -name '*.tbz2' \
    -o -name '*.tar.xz' \
    -o -name '*.txz' \) \
    -print0
)

while IFS= read -r -d '' candidate; do
  install_onnx_file "${candidate}"
done < <(find "${download_dir}" -type f -name '*.onnx' -print0)

while IFS= read -r -d '' candidate; do
  install_onnx_file "${candidate}"
done < <(
  find "${models_dir}" \
    -path "${download_dir}" -prune \
    -o -mindepth 2 -type f -name '*.onnx' -print0
)

if [[ ! -f "${scorer_onnx}" || ! -f "${refiner_onnx}" ]]; then
  echo "Downloaded files to ${download_dir}, but did not find both expected ONNX models." >&2
  echo "Expected:" >&2
  echo "  ${scorer_onnx}" >&2
  echo "  ${refiner_onnx}" >&2
  exit 1
fi

echo "ONNX models are ready in ${models_dir}"
