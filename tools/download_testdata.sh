#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_data_url="${FOUNDATIONPOSE_TEST_DATA_URL:-https://drive.google.com/drive/folders/1pRyFmxYXmAnpku7nGRioZaKrVJtIsroP}"
test_data_dir="${FOUNDATIONPOSE_TEST_DATA_DIR:-${repo_root}/test_data}"
download_dir="${FOUNDATIONPOSE_TEST_DATA_DOWNLOAD_DIR:-${test_data_dir}/.download}"
mustard_dir="${test_data_dir}/mustard0"

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

install_mustard_dir() {
  local candidate="$1"

  if [[ -d "${mustard_dir}" ]]; then
    return 0
  fi

  if [[ -d "${candidate}" ]]; then
    mv "${candidate}" "${mustard_dir}"
  fi
}

if ! command -v gdown >/dev/null 2>&1; then
  echo "Missing gdown. Run this through Pixi with: pixi run download-testdata" >&2
  exit 1
fi

if [[ -d "${mustard_dir}" && "${FOUNDATIONPOSE_FORCE_DOWNLOAD:-0}" != "1" ]]; then
  echo "Test data already exists: ${mustard_dir}"
  echo "Set FOUNDATIONPOSE_FORCE_DOWNLOAD=1 to download again."
  exit 0
fi

mkdir -p "${test_data_dir}" "${download_dir}"

echo "Downloading FoundationPose test data"
echo "  from: ${test_data_url}"
echo "  to:   ${test_data_dir}"

gdown --folder --continue -O "${download_dir}/" "${test_data_url}"

while IFS= read -r -d '' archive; do
  echo "Extracting ${archive}"
  extract_archive "${archive}" "${test_data_dir}"
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

install_mustard_dir "${download_dir}/mustard0"

while IFS= read -r -d '' candidate; do
  install_mustard_dir "${candidate}"
done < <(find "${download_dir}" -type d -name mustard0 -print0)

while IFS= read -r -d '' candidate; do
  install_mustard_dir "${candidate}"
done < <(
  find "${test_data_dir}" \
    -path "${download_dir}" -prune \
    -o -mindepth 2 -type d -name mustard0 -print0
)

if [[ ! -d "${mustard_dir}" ]]; then
  echo "Downloaded files to ${download_dir}, but did not find mustard0." >&2
  echo "Move or extract the dataset so ${mustard_dir} exists." >&2
  exit 1
fi

echo "Test data is ready: ${mustard_dir}"
