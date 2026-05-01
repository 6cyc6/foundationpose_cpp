# Non-ROS Code Overview

This document summarizes the repository code outside `ros2/` and `ros2_ws/`.
The project is a C++/CUDA implementation of the FoundationPose 6-DoF pose
pipeline, with a reusable deployment layer from `easy_deploy_tool`.

## Top-Level Layout

```text
.
|-- CMakeLists.txt
|-- detection_6d_foundationpose/
|-- easy_deploy_tool/
|-- simple_tests/
|-- tools/
|-- docs/
|-- models/
`-- test_data/
```

- `CMakeLists.txt` builds three main non-ROS targets:
  - `easy_deploy_tool`
  - `detection_6d_foundationpose`
  - `simple_tests`
- `detection_6d_foundationpose/` contains the actual FoundationPose model wrapper,
  mesh loading, CUDA pose sampling, rendering, and decoder utilities.
- `easy_deploy_tool/` contains common inference abstractions, async pipelines,
  TensorRT/ONNX Runtime/RKNN backends, and image preprocessing/postprocessing helpers.
- `simple_tests/` contains GoogleTest demos and speed tests.
- `tools/` contains environment setup, CMake configuration, TensorRT conversion,
  and demo runner scripts.
- `models/` is expected to contain converted model engines, especially
  `refiner_hwc_dynamic_fp16.engine` and `scorer_hwc_dynamic_fp16.engine`.
- `test_data/` contains local demo datasets and generated demo output.

## Build Structure

The root CMake file only wires subprojects together:

```cmake
add_subdirectory(easy_deploy_tool)
add_subdirectory(detection_6d_foundationpose)
add_subdirectory(simple_tests)
```

Important build flags and options:

- `ENABLE_TENSORRT` enables `easy_deploy_tool/inference_core/trt_core`.
- `ENABLE_ORT` enables `easy_deploy_tool/inference_core/ort_core`.
- `ENABLE_RKNN` enables `easy_deploy_tool/inference_core/rknn_core`.
- `detection_6d_foundationpose` requires CUDA, CV-CUDA, OpenCV, Eigen, glog,
  Assimp, and the `deploy_core` library.
- `pixi.toml` defines a local Linux CUDA 12 development environment and tasks:
  `bootstrap-cvcuda`, `doctor`, `configure`, `build`, `convert-models`, and `demo`.

## FoundationPose Module

Path: `detection_6d_foundationpose/`

Main public headers:

- `include/detection_6d_foundationpose/foundationpose.hpp`
- `include/detection_6d_foundationpose/mesh_loader.hpp`

Main implementation files:

- `src/foundationpose.cpp`
- `src/foundationpose_sampling.cpp`
- `src/foundationpose_sampling.cu`
- `src/foundationpose_render.cpp`
- `src/foundationpose_render.cu`
- `src/foundationpose_decoder.cu`
- `src/foundationpose_utils.cu`
- `src/mesh_loader/assimp_mesh_loader.cpp`

### Public 6-DoF Interface

`Base6DofDetectionModel` is the public abstract interface.

Key methods:

- `Register(rgb, depth, mask, target_name, out_pose_in_mesh, refine_itr)`
  estimates the initial object pose from RGB, depth, mask, mesh, and camera
  intrinsics.
- `Track(rgb, depth, hyp_pose_in_mesh, target_name, out_pose_in_mesh, refine_itr)`
  refines a known or previous pose for later frames.

Factory:

- `CreateFoundationPoseModel(refiner_core, scorer_core, mesh_loaders, intrinsic, max_h, max_w)`
  creates the concrete `FoundationPose` implementation.

### Concrete FoundationPose Flow

The concrete `FoundationPose` class lives inside `src/foundationpose.cpp`.

Constructor responsibilities:

- Stores the refiner and scorer inference cores.
- Verifies that the refiner/scorer buffers contain expected blob names:
  `render_input`, `transf_input`, `trans`, `rot`, and `scores`.
- Registers all mesh loaders by target name.
- Creates one `FoundationPoseRenderer` per target mesh.
- Creates one shared `FoundationPoseSampler`.

`Register` flow:

1. `CheckInputArguments`
   validates RGB/depth/mask dimensions, max image size, and `target_name`.
2. `UploadDataToDevice`
   copies RGB and depth to CUDA memory and converts depth to an XYZ map.
3. `RefinePreProcess`
   generates pose hypotheses if none exist, renders crops for each hypothesis,
   and prepares refiner input tensors.
4. `refiner_core_->SyncInfer`
   runs the refiner model.
5. `RefinePostProcess`
   decodes translation and rotation deltas and updates all hypotheses.
6. Steps 3 to 5 repeat for `refine_itr`.
7. `ScorePreprocess`
   renders all refined hypotheses for the scorer.
8. `scorer_core_->SyncInfer`
   runs the scorer model.
9. `ScorePostProcess`
   picks the hypothesis with the maximum score.
10. The selected pose is returned in mesh coordinates.

`Track` flow:

1. Validates RGB/depth and `target_name`.
2. Starts from the provided `hyp_pose_in_mesh`.
3. Uploads RGB/depth and builds the XYZ map.
4. Runs only the refiner loop.
5. Returns the refined single pose.

Tracking is much lighter than registration because it does not sample a full
rotation grid and does not run scorer selection.

### Mesh Loading

Public interface: `BaseMeshLoader`.

Important methods:

- `GetName`
- `GetMeshDiameter`
- `GetMeshNumVertices`
- `GetMeshNumFaces`
- `GetMeshVertices`
- `GetMeshVertexNormals`
- `GetMeshTextureCoords`
- `GetMeshTriangleFaces`
- `GetMeshModelCenter`
- `GetOrientBounds`
- `GetObjectDimension`
- `GetTextureMap`

Factory:

- `CreateAssimpMeshLoader(name, mesh_file_path)`

Concrete implementation: `AssimpMeshLoader`.

Responsibilities:

- Loads mesh files through Assimp.
- Triangulates faces and joins identical vertices.
- Extracts vertices, normals, UV texture coordinates, and triangle indices.
- Computes mesh diameter.
- Computes a model center from min/max vertices.
- Computes an oriented bounding box using covariance/eigen decomposition.
- Loads the diffuse texture referenced by the material.
- Falls back to a default gray `2x2` texture if no texture is found.

Helper:

- `ConvertPoseMesh2BBox(pose_in_mesh, mesh_loader)` transforms a pose from mesh
  coordinates into the bounding-box coordinate frame used for drawing.

### Pose Sampling

Files:

- `src/foundationpose_sampling.hpp`
- `src/foundationpose_sampling.cpp`
- `src/foundationpose_sampling.cu`

Class: `FoundationPoseSampler`.

Main method:

- `GetHypPoses(depth_on_device, mask_on_host, input_h, input_w, out_hyp_poses)`

Responsibilities:

- Precomputes a rotation grid from an icosphere plus in-plane rotations.
- Uses CUDA depth erosion and bilateral filtering.
- Copies filtered depth back to host.
- Estimates object translation from the mask bounding box center and valid
  masked depth median.
- Combines the estimated translation with all precomputed rotations to produce
  initial pose hypotheses.

Important helper functions:

- `GenerateIcosphere`
- `SampleViewsIcosphere`
- `MakeRotationGrid`
- `RotationGeodesticDistance`
- `ClusterPoses`
- `GuessTranslation`
- CUDA wrappers `erode_depth` and `bilateral_filter_depth`

### Rendering And Crop Preparation

Files:

- `src/foundationpose_render.hpp`
- `src/foundationpose_render.cpp`
- `src/foundationpose_render.cu`

Class: `FoundationPoseRenderer`.

Main method:

- `RenderAndTransform(poses, rgb_on_device, depth_on_device, xyz_map_on_device,
  input_h, input_w, render_buffer, transf_buffer, crop_ratio)`

Responsibilities:

- Loads mesh vertex, normal, face, UV, and texture data into CUDA buffers.
- Computes crop-window transforms for pose hypotheses.
- Builds camera projection matrices from intrinsics.
- Renders mesh color and XYZ maps through `nvdiffrast`/CUDA rasterization.
- Uses CV-CUDA warp/flip/convert operations for crop preparation.
- Builds two model inputs:
  - `render_input`: rendered color plus rendered XYZ data.
  - `transf_input`: observed RGB/depth-derived XYZ data transformed into the
    same crop window.

Important CPU helpers:

- `ComputeTF`
- `ComputeCropWindowTF`
- `TransformPts`
- `ConstructBBox2D`
- `ProjectMatrixFromIntrinsics`
- `WrapImgPtrToNHWCTensor`
- `WrapFloatPtrToNHWCTensor`

Important CUDA wrappers:

- `clamp`
- `threshold_and_downscale_pointcloud`
- `concat`
- `rasterize`
- `interpolate`
- `texture`
- `transform_points`
- `generate_pose_clip`
- `transform_normals`
- `refine_color`

The `src/nvdiffrast/` folder is a local CUDA rasterization dependency used by
the renderer.

### Decoder And Depth Utilities

Files:

- `src/foundationpose_decoder.cu`
- `src/foundationpose_utils.cu`

Functions:

- `getMaxScoreIndex(cuda_stream, scores, N)` returns the index of the best
  scorer output.
- `convert_depth_to_xyz_map(...)` converts a depth map to per-pixel XYZ using
  camera intrinsics and a minimum-depth threshold.

## Easy Deploy Core

Path: `easy_deploy_tool/deploy_core/`

This folder provides the generic runtime layer used by FoundationPose and other
model wrappers.

### Common Types

File: `include/deploy_core/common_defination.h`

- `BBox2D` stores center `x/y`, width, height, confidence, and class.
- `DataLocation` describes whether tensor/image data is on `HOST`, `DEVICE`, or
  unknown.
- `ImageDataFormat` describes `YUV`, `RGB`, `BGR`, or `GRAY`.
- Macros such as `CHECK_STATE` and `MESSURE_DURATION_AND_CHECK_STATE` provide
  error handling and timing logs.

### Tensor Buffers

File: `include/deploy_core/blob_buffer.h`

- `ITensor` is the backend-independent tensor interface.
  - Exposes name, raw pointer, shape, default shape, byte sizes, location
    movement, zero-copy, and deep-copy operations.
- `BlobsTensor` owns a map from blob name to `ITensor`.
  - `GetTensor(name)` retrieves a named tensor.
  - `Reset()` restores all tensors to default shape and host location.

### Blocking Queue

File: `include/deploy_core/block_queue.h`

`BlockQueue<T>` is the producer/consumer queue used by async pipelines and
buffer pools.

Important methods:

- `BlockPush`
- `CoverPush`
- `Take`
- `TryTake`
- `DisablePush`
- `DisableTake`
- `DisableAndClear`
- `SetNoMoreInput`

### Async Pipeline

Files:

- `include/deploy_core/async_pipeline.h`
- `include/deploy_core/async_pipeline_impl.h`

Main classes:

- `IPipelineImageData`
  wraps image memory plus format/location metadata.
- `IPipelinePackage`
  is the base package passed through pipelines and provides `GetInferBuffer`.
- `AsyncPipelineBlock`
  wraps a named function block.
- `AsyncPipelineContext`
  stores a sequence of blocks.
- `PipelineInstance`
  launches one async worker per block plus an output callback thread.
- `BaseAsyncPipeline<ResultType, GenResult>`
  configures named pipelines and returns `std::future<ResultType>` from
  `PushPipeline`.

Typical flow:

```text
package -> preprocess block -> inference context -> postprocess block -> future result
```

### Inference Core Base

Files:

- `include/deploy_core/base_infer_core.h`
- `src/base_infer_core.cpp`

Main classes:

- `IRotInferCore`
  declares `PreProcess`, `Inference`, `PostProcess`, `AllocBlobsBuffer`,
  `GetType`, and `GetName`.
- `MemBufferPool`
  preallocates `BlobsTensor` instances and returns them through shared pointers
  that automatically recycle buffers when released.
- `BaseInferCore`
  provides synchronous inference through `SyncInfer`, exposes async pipeline
  context through `GetPipelineContext`, and owns the memory buffer pool.
- `BaseInferCoreFactory`
  is the backend factory interface.

`BaseInferCore` always has this internal pipeline:

```text
PreProcess -> Inference -> PostProcess
```

### Detection, SAM, And Stereo Bases

Files:

- `include/deploy_core/base_detection.h`
- `include/deploy_core/base_sam.h`
- `include/deploy_core/base_stereo.h`
- matching `.cpp` files in `src/`

`BaseDetectionModel`:

- Defines sync `Detect` and async `DetectAsync`.
- Uses derived-class `PreProcess` and `PostProcess`.
- Inserts the selected inference core pipeline between them.

`BaseSamModel`:

- Supports image encoder plus point decoder and/or box decoder cores.
- Provides sync and async `GenerateMask`.
- Pipeline shapes:
  - box prompt: `ImagePreProcess -> ImageEncoder -> PromptBoxPreProcess -> MaskDecoder -> MaskPostProcess`
  - point prompt: `ImagePreProcess -> ImageEncoder -> PromptPointPreProcess -> MaskDecoder -> MaskPostProcess`

`BaseStereoMatchingModel`:

- Provides sync and async disparity computation.
- Pipeline: `PreProcess -> Inference -> PostProcess`.

`BaseMonoStereoModel`:

- Provides sync and async monocular depth computation.
- Pipeline: `PreProcess -> Inference -> PostProcess`.

`PipelineCvImageWrapper`:

- Adapts `cv::Mat` to `IPipelineImageData`.
- Records dimensions, channel count, data pointer, host location, and BGR/RGB
  format.

## Inference Backends

Path: `easy_deploy_tool/inference_core/`

### TensorRT Backend

Path: `trt_core/`

Public factories:

- `CreateTrtInferCore`
- `CreateTrtInferCoreFactory`

Concrete class: `TrtInferCore`.

Responsibilities:

- Loads a serialized TensorRT engine.
- Resolves input/output tensor names and shapes.
- Supports explicit blob shapes for dynamic models.
- Allocates CUDA-backed `TrtTensor` buffers.
- Moves tensors between host/device during pre/postprocess as needed.
- Runs `enqueueV3` on a per-thread execution context.
- Maintains separate CUDA streams for preprocess, inference, and postprocess.

Utility:

- `src/onnx_to_trt.cpp` builds TensorRT engines from ONNX models.
- `tools/cvt_onnex2trt.bash` wraps conversion for the refiner/scorer models.

### ONNX Runtime Backend

Path: `ort_core/`

Public factories:

- `CreateOrtInferCore`
- `CreateOrtInferCoreFactory`

Concrete class: `OrtInferCore`.

Responsibilities:

- Loads an ONNX model with ONNX Runtime.
- Resolves static input/output tensor shapes, or accepts explicit shapes.
- Allocates host-backed `OrtTensor` buffers.
- Creates ONNX Runtime tensor views and runs `Ort::Session::Run`.
- Uses `PreProcess` and `PostProcess` mainly as tensor pointer preparation
  stages.

### RKNN Backend

Path: `rknn_core/`

Public factories:

- `CreateRknnInferCore`
- `CreateRknnInferCoreFactory`

Concrete class: `RknnInferCore`.

Responsibilities:

- Loads an RKNN model file.
- Creates one or more RKNN contexts for parallel inference.
- Resolves RKNN input/output tensor attributes.
- Allows user-provided input tensor type overrides.
- Allocates host-backed `RknnTensor` buffers.
- Pushes/pops RKNN contexts through `BlockQueue` to control concurrency.

## Image Processing Utilities

Path: `easy_deploy_tool/deploy_utils/image_processing_utils/`

Public header:

- `include/detection_2d_util/detection_2d_util.h`

Factories:

- `CreateCpuDetPreProcess`
- `CreateCudaDetPreProcess`
- `CreateYolov8PostProcessCpuOrigin`
- `CreateYolov8PostProcessCpuTranspose`
- `CreateYolov8PostProcessCpuDivide`

CPU preprocessing:

- Resizes input image with aspect ratio preserved.
- Pads to the target model input size.
- Optionally flips BGR/RGB channel order.
- Optionally normalizes by `mean` and `val`.
- Optionally transposes NHWC image data to CHW layout.
- Returns the resize scale for later bbox restoration.

CUDA preprocessing:

- Provides GPU resize/warp-affine preprocessing for detection models.
- Uses a CUDA warp-affine kernel to place image data into model input layout.

YOLOv8 postprocessing:

- `Origin` handles `[batch, 4 + cls, boxes]`.
- `Transpose` handles `[batch, boxes, 4 + cls]`.
- `Divide` handles a Rockchip/RKNN-style split output with DFL decoding.
- All versions generate `BBox2D` candidates, apply confidence filtering, scale
  boxes back to original image space, and run NMS.

## Simple Tests And Demo

Path: `simple_tests/`

Main file:

- `src/test_foundationpose.cpp`

The test executable links:

- `deploy_core`
- `trt_core`
- `detection_6d_foundationpose`
- OpenCV, glog, and GTest

Important helpers:

- `ResolveProjectRoot`
- `ResolvePathFromEnv`
- `CreateModel`

Environment variables:

- `FOUNDATIONPOSE_MODELS_DIR`
- `FOUNDATIONPOSE_TEST_DATA_DIR`

Expected TensorRT engine names:

- `refiner_hwc_dynamic_fp16.engine`
- `scorer_hwc_dynamic_fp16.engine`

Tests:

- `foundationpose_test.test`
  - Creates refiner/scorer TensorRT cores.
  - Loads camera intrinsics and a mesh.
  - Runs `Register` on the first frame.
  - Draws the 3D bounding box.
  - Runs `Track` over the RGB/depth sequence.
  - Logs the tracking time for each tracked frame in milliseconds.
  - Saves `test_foundationpose_plot.png` and `test_foundationpose_result.mp4`.
- `foundationpose_test.speed_register`
  repeatedly benchmarks registration.
- `foundationpose_test.speed_track`
  registers once, then repeatedly benchmarks tracking.

## Tools

Path: `tools/`

- `check_deps.sh`
  checks CUDA, TensorRT, CV-CUDA, OpenCV, models, test data, and build outputs.
- `cvt_onnx2trt.bash`
  converts ONNX refiner/scorer models to TensorRT engines.
- `run_local_demo.sh`
  verifies engines and dataset, sets library paths, and runs
  `simple_tests --gtest_filter=foundationpose_test.test`.

## Data Flow Summary

The non-ROS FoundationPose runtime can be read as this pipeline:

```text
RGB + depth + mask + mesh + camera K
        |
        v
FoundationPose::Register
        |
        |-- UploadDataToDevice
        |-- convert_depth_to_xyz_map
        |-- FoundationPoseSampler::GetHypPoses
        |     |-- erode_depth
        |     |-- bilateral_filter_depth
        |     `-- GuessTranslation
        |
        |-- FoundationPoseRenderer::RenderAndTransform
        |     |-- render mesh from each hypothesis
        |     |-- crop rendered and observed inputs
        |     `-- fill render_input/transf_input
        |
        |-- refiner TensorRT inference
        |-- RefinePostProcess
        |
        |-- scorer TensorRT inference
        |-- getMaxScoreIndex
        |
        v
pose in mesh frame
```

Tracking is the same refinement path, but it starts from an existing pose and
returns after refiner postprocess:

```text
RGB + depth + previous pose + mesh + camera K
        |
        v
FoundationPose::Track
        |
        |-- UploadDataToDevice
        |-- FoundationPoseRenderer::RenderAndTransform
        |-- refiner TensorRT inference
        `-- RefinePostProcess
        |
        v
updated pose in mesh frame
```

## Extension Points

- Add another mesh format by implementing `BaseMeshLoader`.
- Add another inference runtime by deriving from `BaseInferCore`.
- Add a new 2D detector by deriving from `BaseDetectionModel` and reusing
  detection preprocess/postprocess factories.
- Add a SAM variant by deriving from `BaseSamModel`.
- Add stereo or monocular depth models by deriving from `BaseStereoMatchingModel`
  or `BaseMonoStereoModel`.
- Add another FoundationPose target object by creating another `BaseMeshLoader`
  with a unique `GetName()` and passing it to `CreateFoundationPoseModel`.

## Practical Notes

- RGB inputs to FoundationPose are expected to be RGB, while OpenCV `imread`
  returns BGR by default.
- Depth input is expected as `CV_32FC1`.
- Mask input for registration is expected as `CV_8UC1` with object pixels
  greater than zero.
- Camera intrinsics must match the original image size. If images are resized,
  intrinsics need corresponding adjustment.
- `Register` requires a valid mask. `Track` does not.
- Tensor names are hard-coded for FoundationPose:
  - inputs: `render_input`, `transf_input`
  - refiner outputs: `trans`, `rot`
  - scorer output: `scores`
- The current demo path is TensorRT-oriented. ONNX Runtime and RKNN exist in
  `easy_deploy_tool`, but `simple_tests` uses TensorRT.
