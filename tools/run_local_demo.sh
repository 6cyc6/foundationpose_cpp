#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pixi_prefix="${CONDA_PREFIX:-${repo_root}/.pixi/envs/default}"
build_dir="${FOUNDATIONPOSE_BUILD_DIR:-${repo_root}/build-pixi}"
models_dir="${FOUNDATIONPOSE_MODELS_DIR:-${repo_root}/models}"
test_data_dir="${FOUNDATIONPOSE_TEST_DATA_DIR:-${repo_root}/test_data}"
cvcuda_root="${FOUNDATIONPOSE_CVCUDA_ROOT:-${pixi_prefix}}"
binary_path="${build_dir}/bin/simple_tests"

prepend_path() {
  local var_name="$1"
  local candidate="$2"
  local current_value="${!var_name-}"

  if [[ -z "${candidate}" || ! -e "${candidate}" ]]; then
    return 0
  fi

  case ":${current_value}:" in
    *":${candidate}:"*) ;;
    *)
      if [[ -n "${current_value}" ]]; then
        export "${var_name}=${candidate}:${current_value}"
      else
        export "${var_name}=${candidate}"
      fi
      ;;
  esac
}

detect_tensorrt_lib_dir() {
  local tensorrt_root="${TENSORRT_ROOT:-}"
  local candidates=()
  local candidate

  if [[ -n "${tensorrt_root}" ]]; then
    candidates+=(
      "${tensorrt_root}/targets/x86_64-linux/lib"
      "${tensorrt_root}/lib"
      "${tensorrt_root}/lib64"
    )
  fi

  candidates+=(
    "${pixi_prefix}/lib"
    "${pixi_prefix}/targets/x86_64-linux/lib"
    "/usr/local/cuda/targets/x86_64-linux/lib"
    /usr/local/cuda-*/targets/x86_64-linux/lib
    "/usr/lib/x86_64-linux-gnu"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/libnvinfer.so" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

if [[ ! -x "${binary_path}" ]]; then
  echo "Missing ${binary_path}. Build the project first." >&2
  exit 1
fi

if [[ ! -f "${models_dir}/refiner_hwc_dynamic_fp16.engine" || ! -f "${models_dir}/scorer_hwc_dynamic_fp16.engine" ]]; then
  echo "Missing TensorRT engine files in ${models_dir}." >&2
  exit 1
fi

if [[ ! -d "${test_data_dir}/mustard0" ]]; then
  echo "Missing demo dataset in ${test_data_dir}/mustard0." >&2
  exit 1
fi

tensorrt_lib_dir="$(detect_tensorrt_lib_dir || true)"
if [[ -z "${tensorrt_lib_dir}" ]]; then
  echo "TensorRT not found. Please install it or export your TENSORRT_ROOT." >&2
  exit 1
fi

prepend_path LD_LIBRARY_PATH "${cvcuda_root}/lib/x86_64-linux-gnu"
prepend_path LD_LIBRARY_PATH "${cvcuda_root}/lib"
prepend_path LD_LIBRARY_PATH "${tensorrt_lib_dir}"
prepend_path LD_LIBRARY_PATH "${build_dir}/lib"

exec "${binary_path}" --gtest_filter=foundationpose_test.test
