#include "tests/fps_counter.h"
#include "tests/help_func.hpp"
#include "tests/fs_util.hpp"

#include "detection_6d_foundationpose/foundationpose.hpp"
#include "trt_core/trt_core.h"

#include <chrono>
#include <cstdlib>
#include <gtest/gtest.h>

using namespace inference_core;
using namespace detection_6d;

#ifndef DEFAULT_FOUNDATIONPOSE_MODELS_DIR
#define DEFAULT_FOUNDATIONPOSE_MODELS_DIR "models"
#endif

#ifndef DEFAULT_FOUNDATIONPOSE_TEST_DATA_DIR
#define DEFAULT_FOUNDATIONPOSE_TEST_DATA_DIR "test_data"
#endif

namespace
{

std::string EnvOrDefault(const char* name, const char* default_value)
{
  const char* value = std::getenv(name);
  return (value != nullptr && value[0] != '\0') ? std::string(value) : std::string(default_value);
}

std::string JoinPath(const std::string& base, const std::string& relative)
{
  if (base.empty() || base.back() == '/')
  {
    return base + relative;
  }

  return base + "/" + relative;
}

}  // namespace

static const std::string models_path_         = EnvOrDefault("FOUNDATIONPOSE_MODELS_DIR", DEFAULT_FOUNDATIONPOSE_MODELS_DIR);
static const std::string test_data_path_      = EnvOrDefault("FOUNDATIONPOSE_TEST_DATA_DIR", DEFAULT_FOUNDATIONPOSE_TEST_DATA_DIR);
static const std::string refiner_engine_path_ = JoinPath(models_path_, "refiner_hwc_dynamic_fp16.engine");
static const std::string scorer_engine_path_  = JoinPath(models_path_, "scorer_hwc_dynamic_fp16.engine");
static const std::string demo_data_path_      = JoinPath(test_data_path_, "mustard0");
static const std::string demo_textured_obj_path = demo_data_path_ + "/mesh/textured_simple.obj";
static const std::string demo_textured_map_path = demo_data_path_ + "/mesh/texture_map.png";
static const std::string demo_name_             = "mustard";
static const std::string frame_id               = "1581120424100262102";
// static const size_t      refine_itr             = 1;
static const size_t      refine_itr             = 5;
static const std::string register_plot_path_     = JoinPath(test_data_path_, "test_foundationpose_plot.png");
static const std::string result_video_path_      = JoinPath(test_data_path_, "test_foundationpose_result.mp4");

std::tuple<std::shared_ptr<Base6DofDetectionModel>, std::shared_ptr<BaseMeshLoader>> CreateModel()
{
  auto refiner_core = CreateTrtInferCore(refiner_engine_path_,
                                         {
                                             {"transf_input", {252, 160, 160, 6}},
                                             {"render_input", {252, 160, 160, 6}},
                                         },
                                         {{"trans", {252, 3}}, {"rot", {252, 3}}}, 1);
  auto scorer_core  = CreateTrtInferCore(scorer_engine_path_,
                                         {
                                            {"transf_input", {252, 160, 160, 6}},
                                            {"render_input", {252, 160, 160, 6}},
                                        },
                                         {{"scores", {252, 1}}}, 1);

  Eigen::Matrix3f intrinsic_in_mat = ReadCamK(demo_data_path_ + "/cam_K.txt");

  auto mesh_loader = CreateAssimpMeshLoader(demo_name_, demo_textured_obj_path);
  CHECK(mesh_loader != nullptr);

  auto foundation_pose =
      CreateFoundationPoseModel(refiner_core, scorer_core, {mesh_loader}, intrinsic_in_mat);

  return {foundation_pose, mesh_loader};
}

TEST(foundationpose_test, test)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  const Eigen::Vector3f object_dimension = mesh_loader->GetObjectDimension();

  Eigen::Matrix4f out_pose;
  CHECK(foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose, refine_itr));
  LOG(WARNING) << "first Pose : " << out_pose;

  // [temp] for test
  cv::Mat regist_plot = rgb.clone();
  cv::cvtColor(regist_plot, regist_plot, cv::COLOR_RGB2BGR);
  auto draw_pose = ConvertPoseMesh2BBox(out_pose, mesh_loader);
  draw3DBoundingBox(intrinsic_in_mat, draw_pose, 480, 640, object_dimension, regist_plot);
  cv::imwrite(register_plot_path_, regist_plot);

  auto rgb_paths = get_files_in_directory(demo_data_path_ + "/rgb/");
  std::sort(rgb_paths.begin(), rgb_paths.end());
  std::vector<std::string> frame_ids;
  for (const auto &rgb_path : rgb_paths)
  {
    frame_ids.push_back(rgb_path.stem());
  }

  int                  total = frame_ids.size();
  std::vector<cv::Mat> result_image_sequence{regist_plot};
  for (int i = 1; i < total; ++i)
  {
    std::string cur_rgb_path   = demo_data_path_ + "/rgb/" + frame_ids[i] + ".png";
    std::string cur_depth_path = demo_data_path_ + "/depth/" + frame_ids[i] + ".png";
    auto [cur_rgb, cur_depth]  = ReadRgbDepth(cur_rgb_path, cur_depth_path);

    Eigen::Matrix4f track_pose;
    const auto      track_start = std::chrono::steady_clock::now();
    // CHECK(foundation_pose->Track(cur_rgb.clone(), cur_depth, out_pose, demo_name_, track_pose));
    CHECK(foundation_pose->Track(cur_rgb.clone(), cur_depth, out_pose, demo_name_, track_pose, refine_itr));
    const auto track_end = std::chrono::steady_clock::now();
    const auto track_ms =
        std::chrono::duration<double, std::milli>(track_end - track_start).count();
    LOG(WARNING) << "Tracking iteration " << i << "/" << (total - 1) << " time: " << track_ms
                 << " ms";
    LOG(WARNING) << "Track pose : " << track_pose;

    cv::Mat track_plot = cur_rgb.clone();
    cv::cvtColor(track_plot, track_plot, cv::COLOR_RGB2BGR);
    auto draw_pose = ConvertPoseMesh2BBox(track_pose, mesh_loader);
    draw3DBoundingBox(intrinsic_in_mat, draw_pose, 480, 640, object_dimension, track_plot);
    cv::imshow("test_foundationpose_result", track_plot);
    cv::waitKey(20);
    result_image_sequence.push_back(track_plot);

    out_pose = track_pose;
  }

  saveVideo(result_image_sequence, result_video_path_);
}

TEST(foundationpose_test, speed_register)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0; i < 50; ++i)
  {
    Eigen::Matrix4f out_pose;
    foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, out_pose);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();
}

TEST(foundationpose_test, speed_track)
{
  auto [foundation_pose, mesh_loader] = CreateModel();
  Eigen::Matrix3f intrinsic_in_mat    = ReadCamK(demo_data_path_ + "/cam_K.txt");

  const std::string first_rgb_path   = demo_data_path_ + "/rgb/" + frame_id + ".png";
  const std::string first_depth_path = demo_data_path_ + "/depth/" + frame_id + ".png";
  const std::string first_mask_path  = demo_data_path_ + "/masks/" + frame_id + ".png";

  auto [rgb, depth, mask] = ReadRgbDepthMask(first_rgb_path, first_depth_path, first_mask_path);

  Eigen::Matrix4f first_pose;
  foundation_pose->Register(rgb.clone(), depth, mask, demo_name_, first_pose);

  // proccess
  FPSCounter counter;
  counter.Start();
  for (int i = 0; i < 5000; ++i)
  {
    Eigen::Matrix4f track_pose;
    // foundation_pose->Track(rgb.clone(), depth, first_pose, demo_name_, track_pose);
    foundation_pose->Track(rgb.clone(), depth, first_pose, demo_name_, track_pose, refine_itr);
    counter.Count(1);
  }

  LOG(WARNING) << "average fps: " << counter.GetFPS();
}
