#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pixi_prefix="${CONDA_PREFIX:-${repo_root}/.pixi/envs/default}"
models_dir="${FOUNDATIONPOSE_MODELS_DIR:-${repo_root}/models}"
build_dir="${FOUNDATIONPOSE_BUILD_DIR:-${repo_root}/build-pixi}"
onnx_to_trt_bin="${FOUNDATIONPOSE_ONNX_TO_TRT_BIN:-${build_dir}/bin/onnx_to_trt}"

scorer_onnx="${models_dir}/scorer_hwc.onnx"
refiner_onnx="${models_dir}/refiner_hwc.onnx"
scorer_engine="${models_dir}/scorer_hwc_dynamic_fp16.engine"
refiner_engine="${models_dir}/refiner_hwc_dynamic_fp16.engine"

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

detect_tensorrt_root() {
  local candidates=(
    "${TENSORRT_ROOT:-}"
    "${pixi_prefix}"
    "/usr"
    "/usr/local/cuda"
    /usr/local/cuda-*
  )
  local candidate

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && \
          ( -f "${candidate}/include/NvInfer.h" || \
            -f "${candidate}/include/x86_64-linux-gnu/NvInfer.h" || \
            -f "${candidate}/targets/x86_64-linux/include/NvInfer.h" ) ]]; then
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
      "${tensorrt_root}/lib/x86_64-linux-gnu"
      "${tensorrt_root}/targets/x86_64-linux/lib"
      "${tensorrt_root}/lib"
      "${tensorrt_root}/lib64"
    )
  fi

  candidates+=(
    "${pixi_prefix}/lib"
    "${pixi_prefix}/targets/x86_64-linux/lib"
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

if [[ ! -f "${scorer_onnx}" ]]; then
  echo "Missing ONNX model: ${scorer_onnx}" >&2
  echo "Run: pixi run download-onnx-model" >&2
  exit 1
fi

if [[ ! -f "${refiner_onnx}" ]]; then
  echo "Missing ONNX model: ${refiner_onnx}" >&2
  echo "Run: pixi run download-onnx-model" >&2
  exit 1
fi

tensorrt_root="$(detect_tensorrt_root || true)"
if [[ -z "${tensorrt_root}" ]]; then
  echo "TensorRT not found. Please install it or export your TENSORRT_ROOT." >&2
  exit 1
fi

tensorrt_lib_dir="$(detect_tensorrt_lib_dir "${tensorrt_root}" || true)"
if [[ -z "${tensorrt_lib_dir}" ]]; then
  echo "TensorRT not found. Please install it or export your TENSORRT_ROOT." >&2
  exit 1
fi

export TENSORRT_ROOT="${tensorrt_root}"
prepend_path LD_LIBRARY_PATH "${tensorrt_lib_dir}"
prepend_path LD_LIBRARY_PATH "${pixi_prefix}/lib"

convert_with() {
  local converter="$1"
  local onnx_path="$2"
  local engine_path="$3"

  "${converter}" --onnx="${onnx_path}" \
                 --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
                 --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                 --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
                 --fp16 \
                 --saveEngine="${engine_path}"
}

trtexec_bin="$(detect_trtexec "${tensorrt_root}" || true)"
if [[ -n "${trtexec_bin}" ]]; then
  export TRTEXEC_BIN="${trtexec_bin}"
  converter="${trtexec_bin}"
else
  if [[ ! -x "${onnx_to_trt_bin}" ]]; then
    echo "TensorRT trtexec not found, and fallback converter is missing: ${onnx_to_trt_bin}" >&2
    echo "Run: pixi run build" >&2
    echo "Or install TensorRT tools and export TRTEXEC_BIN." >&2
    echo "TENSORRT_ROOT was detected as: ${tensorrt_root}" >&2
    exit 1
  fi
  converter="${onnx_to_trt_bin}"
fi

echo "Converting ONNX models with ${converter}"
echo "  models: ${models_dir}"

convert_with "${converter}" "${scorer_onnx}" "${scorer_engine}"
convert_with "${converter}" "${refiner_onnx}" "${refiner_engine}"

echo "TensorRT engines are ready:"
echo "  ${scorer_engine}"
echo "  ${refiner_engine}"
