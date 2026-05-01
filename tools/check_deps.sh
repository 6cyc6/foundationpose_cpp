#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pixi_prefix="${CONDA_PREFIX:-${repo_root}/.pixi/envs/default}"
build_dir="${FOUNDATIONPOSE_BUILD_DIR:-${repo_root}/build-pixi}"
models_dir="${FOUNDATIONPOSE_MODELS_DIR:-${repo_root}/models}"
test_data_dir="${FOUNDATIONPOSE_TEST_DATA_DIR:-${repo_root}/test_data}"
cvcuda_root="${FOUNDATIONPOSE_CVCUDA_ROOT:-${pixi_prefix}}"
unwind_include_dir="${FOUNDATIONPOSE_UNWIND_INCLUDE_DIR:-${pixi_prefix}/include}"
onnx_to_trt_bin="${FOUNDATIONPOSE_ONNX_TO_TRT_BIN:-${build_dir}/bin/onnx_to_trt}"

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

detect_cuda_root() {
  local candidates=(
    "${FOUNDATIONPOSE_CUDA_ROOT:-}"
    "${pixi_prefix}"
    "${CUDA_HOME:-}"
    "${CUDA_PATH:-}"
  )
  local candidate

  if [[ "${FOUNDATIONPOSE_ALLOW_SYSTEM_DEPS:-1}" != "0" ]]; then
    candidates+=("/usr/local/cuda" /usr/local/cuda-*)
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

detect_tensorrt_root() {
  local candidates=(
    "${TENSORRT_ROOT:-}"
    "${pixi_prefix}"
  )
  local candidate

  if [[ "${FOUNDATIONPOSE_ALLOW_SYSTEM_DEPS:-1}" != "0" ]]; then
    candidates+=("/usr/local/cuda" /usr/local/cuda-* "/usr")
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && \
          ( -f "${candidate}/include/NvInfer.h" || -f "${candidate}/targets/x86_64-linux/include/NvInfer.h" ) ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

detect_tensorrt_lib_dir() {
  local tensorrt_root="$1"
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
  )

  if [[ "${FOUNDATIONPOSE_ALLOW_SYSTEM_DEPS:-1}" != "0" ]]; then
    candidates+=(
      "/usr/local/cuda/targets/x86_64-linux/lib"
      /usr/local/cuda-*/targets/x86_64-linux/lib
      "/usr/lib/x86_64-linux-gnu"
    )
  fi

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}/libnvinfer.so" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

detect_trtexec() {
  local tensorrt_root="$1"
  local candidates=("${TRTEXEC_BIN:-}")
  local command_trtexec
  local candidate

  command_trtexec="$(command -v trtexec 2>/dev/null || true)"
  if [[ -n "${command_trtexec}" ]]; then
    candidates+=("${command_trtexec}")
  fi

  if [[ -n "${tensorrt_root}" ]]; then
    candidates+=(
      "${tensorrt_root}/bin/trtexec"
      "${tensorrt_root}/targets/x86_64-linux/bin/trtexec"
    )
  fi

  candidates+=(
    "/usr/src/tensorrt/bin/trtexec"
    "/usr/local/bin/trtexec"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  return 1
}

has_working_nvidia_driver() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1
}

status_line() {
  local label="$1"
  local value="$2"
  printf '%-20s %s\n' "${label}" "${value}"
}

path_status() {
  local label="$1"
  local path="$2"
  if [[ -e "${path}" ]]; then
    status_line "${label}" "ok: ${path}"
  else
    status_line "${label}" "missing: ${path}"
  fi
}

cuda_root="$(detect_cuda_root || true)"
tensorrt_root="$(detect_tensorrt_root || true)"
tensorrt_lib_dir="$(detect_tensorrt_lib_dir "${tensorrt_root}" || true)"
trtexec_bin="$(detect_trtexec "${tensorrt_root}" || true)"
if [[ -n "${trtexec_bin}" ]]; then
  model_converter_status="ok: ${trtexec_bin} (trtexec)"
elif [[ -x "${onnx_to_trt_bin}" ]]; then
  model_converter_status="ok: ${onnx_to_trt_bin} (fallback)"
else
  model_converter_status="missing: run pixi run build or install TensorRT tools"
fi

if [[ -n "${cuda_root}" ]]; then
  export CUDA_HOME="${cuda_root}"
  export CUDA_PATH="${cuda_root}"
  export CUDAToolkit_ROOT="${cuda_root}"
  export CUDA_TOOLKIT_ROOT_DIR="${cuda_root}"
  prepend_path PATH "${cuda_root}/bin"
  prepend_path LD_LIBRARY_PATH "${cuda_root}/targets/x86_64-linux/lib"
  prepend_path LD_LIBRARY_PATH "${cuda_root}/lib64"
  prepend_path LD_LIBRARY_PATH "${cuda_root}/lib"
fi

if [[ -n "${tensorrt_root}" ]]; then
  export TENSORRT_ROOT="${tensorrt_root}"
fi
if [[ -n "${tensorrt_lib_dir}" ]]; then
  prepend_path LD_LIBRARY_PATH "${tensorrt_lib_dir}"
fi
if [[ -n "${trtexec_bin}" ]]; then
  export TRTEXEC_BIN="${trtexec_bin}"
  prepend_path PATH "$(dirname "${trtexec_bin}")"
fi

echo "FoundationPose local dependencies"
echo
status_line "Repo root" "${repo_root}"
status_line "Build dir" "${build_dir}"
status_line "Models dir" "${models_dir}"
status_line "Test data dir" "${test_data_dir}"
echo
status_line "CUDA_HOME" "${CUDA_HOME:-missing}"
status_line "TENSORRT_ROOT" "${TENSORRT_ROOT:-missing}"
status_line "TensorRT lib" "${tensorrt_lib_dir:-missing}"
if [[ -n "${TRTEXEC_BIN:-}" ]]; then
  status_line "TRTEXEC_BIN" "${TRTEXEC_BIN}"
elif [[ -x "${onnx_to_trt_bin}" ]]; then
  status_line "TRTEXEC_BIN" "optional: missing (using onnx_to_trt fallback)"
else
  status_line "TRTEXEC_BIN" "missing"
fi
status_line "Model converter" "${model_converter_status}"
status_line "Unwind include" "${unwind_include_dir}"
if has_working_nvidia_driver; then
  status_line "NVIDIA driver" "ok"
else
  status_line "NVIDIA driver" "missing or inaccessible"
fi
echo
path_status "CUDA nvcc" "${CUDA_HOME:-}/bin/nvcc"
path_status "CV-CUDA root" "${cvcuda_root}"
if [[ -f "${cvcuda_root}/lib/cmake/nvcv_types/nvcv_types-config.cmake" ]]; then
  path_status "CV-CUDA cmake" "${cvcuda_root}/lib/cmake/nvcv_types/nvcv_types-config.cmake"
else
  path_status "CV-CUDA cmake" "${cvcuda_root}/lib/x86_64-linux-gnu/cmake/nvcv_types/nvcv_types-config.cmake"
fi
path_status "OpenCV cmake" "${pixi_prefix}/lib/cmake/opencv4/OpenCVConfig.cmake"
path_status "TensorRT header" "${TENSORRT_ROOT:-}/include/NvInfer.h"
path_status "simple_tests" "${build_dir}/bin/simple_tests"
path_status "onnx_to_trt" "${onnx_to_trt_bin}"
path_status "scorer ONNX" "${models_dir}/scorer_hwc.onnx"
path_status "refiner ONNX" "${models_dir}/refiner_hwc.onnx"
path_status "Scorer engine" "${models_dir}/scorer_hwc_dynamic_fp16.engine"
path_status "Refiner engine" "${models_dir}/refiner_hwc_dynamic_fp16.engine"
path_status "Mustard dataset" "${test_data_dir}/mustard0"
