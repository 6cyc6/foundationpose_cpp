#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace
{

class Logger final : public nvinfer1::ILogger
{
 public:
  void log(Severity severity, const char* msg) noexcept override
  {
    if (severity <= Severity::kWARNING)
    {
      std::cerr << "[TensorRT] " << msg << '\n';
    }
  }
};

template <typename T>
void destroy_trt_object(T* object)
{
  if (object == nullptr)
  {
    return;
  }

#if NV_TENSORRT_MAJOR >= 10
  delete object;
#else
  object->destroy();
#endif
}

template <typename T>
using TrtPtr = std::unique_ptr<T, decltype(&destroy_trt_object<T>)>;

template <typename T>
TrtPtr<T> make_trt_ptr(T* object)
{
  return TrtPtr<T>(object, &destroy_trt_object<T>);
}

struct Options
{
  std::string onnx_path;
  std::string engine_path;
  std::string min_shapes;
  std::string opt_shapes;
  std::string max_shapes;
  bool        fp16 = false;
};

void print_usage(const char* argv0)
{
  std::cerr << "Usage: " << argv0
            << " --onnx=<model.onnx> --saveEngine=<model.engine>"
               " --minShapes=name:dims[,name:dims]"
               " --optShapes=name:dims[,name:dims]"
               " --maxShapes=name:dims[,name:dims]"
               " [--fp16]\n";
}

bool starts_with(const std::string& value, const std::string& prefix)
{
  return value.rfind(prefix, 0) == 0;
}

std::string value_after_equals(const std::string& argument, const std::string& key)
{
  const std::string prefix = key + "=";
  if (starts_with(argument, prefix))
  {
    return argument.substr(prefix.size());
  }

  return {};
}

bool parse_args(int argc, char** argv, Options& options)
{
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];

    if (arg == "--fp16")
    {
      options.fp16 = true;
    }
    else if (starts_with(arg, "--onnx="))
    {
      options.onnx_path = value_after_equals(arg, "--onnx");
    }
    else if (arg == "--onnx" && i + 1 < argc)
    {
      options.onnx_path = argv[++i];
    }
    else if (starts_with(arg, "--saveEngine="))
    {
      options.engine_path = value_after_equals(arg, "--saveEngine");
    }
    else if (arg == "--saveEngine" && i + 1 < argc)
    {
      options.engine_path = argv[++i];
    }
    else if (starts_with(arg, "--minShapes="))
    {
      options.min_shapes = value_after_equals(arg, "--minShapes");
    }
    else if (starts_with(arg, "--optShapes="))
    {
      options.opt_shapes = value_after_equals(arg, "--optShapes");
    }
    else if (starts_with(arg, "--maxShapes="))
    {
      options.max_shapes = value_after_equals(arg, "--maxShapes");
    }
    else
    {
      std::cerr << "Unknown argument: " << arg << '\n';
      return false;
    }
  }

  return !options.onnx_path.empty() && !options.engine_path.empty() && !options.min_shapes.empty()
         && !options.opt_shapes.empty() && !options.max_shapes.empty();
}

std::vector<std::string> split(const std::string& value, char delimiter)
{
  std::vector<std::string> parts;
  std::stringstream        stream(value);
  std::string              part;

  while (std::getline(stream, part, delimiter))
  {
    if (!part.empty())
    {
      parts.push_back(part);
    }
  }

  return parts;
}

nvinfer1::Dims parse_dims(const std::string& value)
{
  nvinfer1::Dims dims{};
  const auto     parts = split(value, 'x');

  dims.nbDims = static_cast<int32_t>(parts.size());
  for (int32_t i = 0; i < dims.nbDims; ++i)
  {
    dims.d[i] = std::stoi(parts[static_cast<std::size_t>(i)]);
  }

  return dims;
}

std::map<std::string, nvinfer1::Dims> parse_shape_map(const std::string& value)
{
  std::map<std::string, nvinfer1::Dims> shapes;

  for (const auto& entry : split(value, ','))
  {
    const auto separator = entry.find(':');
    if (separator == std::string::npos)
    {
      throw std::runtime_error("Invalid shape entry: " + entry);
    }

    shapes.emplace(entry.substr(0, separator), parse_dims(entry.substr(separator + 1)));
  }

  return shapes;
}

bool set_profile_shapes(nvinfer1::IOptimizationProfile& profile,
                        const std::string&              min_shapes,
                        const std::string&              opt_shapes,
                        const std::string&              max_shapes)
{
  const auto mins = parse_shape_map(min_shapes);
  const auto opts = parse_shape_map(opt_shapes);
  const auto maxs = parse_shape_map(max_shapes);

  for (const auto& [name, min_dims] : mins)
  {
    const auto opt_iter = opts.find(name);
    const auto max_iter = maxs.find(name);

    if (opt_iter == opts.end() || max_iter == maxs.end())
    {
      std::cerr << "Missing opt/max shape for input: " << name << '\n';
      return false;
    }

    if (!profile.setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims)
        || !profile.setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_iter->second)
        || !profile.setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_iter->second))
    {
      std::cerr << "Failed to set optimization profile dimensions for input: " << name << '\n';
      return false;
    }
  }

  return true;
}

bool write_engine(const std::string& path, const nvinfer1::IHostMemory& engine)
{
  std::ofstream output(path, std::ios::binary);
  if (!output)
  {
    std::cerr << "Failed to open engine output path: " << path << '\n';
    return false;
  }

  output.write(static_cast<const char*>(engine.data()), static_cast<std::streamsize>(engine.size()));
  return static_cast<bool>(output);
}

bool has_cuda_device()
{
  int         device_count = 0;
  const auto status       = cudaGetDeviceCount(&device_count);

  if (status != cudaSuccess || device_count <= 0)
  {
    std::cerr << "CUDA-capable device not available. TensorRT engine conversion requires an NVIDIA GPU "
                 "and a working NVIDIA driver.\n";
    std::cerr << "CUDA error: " << cudaGetErrorString(status) << '\n';
    return false;
  }

  return true;
}

}  // namespace

int run_converter(int argc, char** argv)
{
  Options options;
  if (!parse_args(argc, argv, options))
  {
    print_usage(argv[0]);
    return 2;
  }

  Logger logger;
  if (!has_cuda_device())
  {
    return 1;
  }

  initLibNvInferPlugins(&logger, "");

  auto builder = make_trt_ptr(nvinfer1::createInferBuilder(logger));
  if (!builder)
  {
    std::cerr << "Failed to create TensorRT builder.\n";
    return 1;
  }

  const auto network_flags =
      1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = make_trt_ptr(builder->createNetworkV2(network_flags));
  auto config  = make_trt_ptr(builder->createBuilderConfig());
  auto parser  = make_trt_ptr(nvonnxparser::createParser(*network, logger));
  auto profile = builder->createOptimizationProfile();

  if (!network || !config || !profile || !parser)
  {
    std::cerr << "Failed to create TensorRT parser/build objects.\n";
    return 1;
  }

  if (!parser->parseFromFile(options.onnx_path.c_str(),
                             static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING)))
  {
    std::cerr << "Failed to parse ONNX model: " << options.onnx_path << '\n';
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
      std::cerr << parser->getError(i)->desc() << '\n';
    }
    return 1;
  }

  if (!set_profile_shapes(*profile, options.min_shapes, options.opt_shapes, options.max_shapes))
  {
    return 1;
  }

  config->addOptimizationProfile(profile);
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, std::size_t{2} << 30);

  if (options.fp16)
  {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  std::cout << "Building TensorRT engine:\n"
            << "  onnx:   " << options.onnx_path << '\n'
            << "  engine: " << options.engine_path << '\n';

  auto serialized_engine = make_trt_ptr(builder->buildSerializedNetwork(*network, *config));
  if (!serialized_engine)
  {
    std::cerr << "TensorRT failed to build serialized engine.\n";
    return 1;
  }

  if (!write_engine(options.engine_path, *serialized_engine))
  {
    return 1;
  }

  return 0;
}

int main(int argc, char** argv)
{
  try
  {
    return run_converter(argc, argv);
  }
  catch (const std::exception& error)
  {
    std::cerr << "TensorRT conversion failed: " << error.what() << '\n';
    return 1;
  }
}
