#include "tensor.h"
#include "storage.h"
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>

namespace dense {

namespace {

thread_local std::mt19937 generator(std::random_device{}());
thread_local std::normal_distribution<double> normal_dist(0.0, 1.0); //标准正态分布
thread_local std::uniform_real_distribution<> uniform_real_dis(0, 1); //均匀分布

std::vector<size_t> calculate_stride(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<size_t> stride(shape.size());
  stride[shape.size() - 1] = 1; // 最后一个维度的步长为1

  // 从倒数第二个维度开始，向前计算
  for (int i = shape.size() - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }

  return stride;
}

// 通用类型分发辅助函数：根据 DType 将数据指针转换为正确类型，并执行填充函数
template <typename F>
void dispatch_fill_by_dtype(DType dtype, uint8_t *data_ptr, size_t num_elements,
                            F fill_func) {
  switch (dtype) {
  case kBool:
    fill_func(reinterpret_cast<bool *>(data_ptr), num_elements);
    break;
  case kUInt8:
    fill_func(reinterpret_cast<uint8_t *>(data_ptr), num_elements);
    break;
  case kInt8:
    fill_func(reinterpret_cast<int8_t *>(data_ptr), num_elements);
    break;
  case kUInt16:
    fill_func(reinterpret_cast<uint16_t *>(data_ptr), num_elements);
    break;
  case kInt16:
    fill_func(reinterpret_cast<int16_t *>(data_ptr), num_elements);
    break;
  case kFloat16: // 暂时作为 uint16_t 处理，内部lambda根据类型再做转换
  case kBFloat16:
    fill_func(reinterpret_cast<uint16_t *>(data_ptr), num_elements);
    break;
  case kUInt32:
    fill_func(reinterpret_cast<uint32_t *>(data_ptr), num_elements);
    break;
  case kInt32:
    fill_func(reinterpret_cast<int32_t *>(data_ptr), num_elements);
    break;
  case kFloat32:
    fill_func(reinterpret_cast<float *>(data_ptr), num_elements);
    break;
  case kUInt64:
    fill_func(reinterpret_cast<uint64_t *>(data_ptr), num_elements);
    break;
  case kInt64:
    fill_func(reinterpret_cast<int64_t *>(data_ptr), num_elements);
    break;
  case kFloat64:
    fill_func(reinterpret_cast<double *>(data_ptr), num_elements);
    break;
  default:
    // 默认情况下填充零
    std::memset(data_ptr, 0, num_elements * get_element_size(dtype));
    break;
  }
}
} // namespace

DType dtype_from_string(const std::string &v) {
  static const std::unordered_map<std::string, DType> dtype_map = {
      {"BOOL", DType::kBool},     {"U8", DType::kUInt8},
      {"I8", DType::kInt8},       {"U16", DType::kUInt16},
      {"I16", DType::kInt16},     {"U32", DType::kUInt32},
      {"I32", DType::kInt32},     {"U64", DType::kUInt64},
      {"I64", DType::kInt64},     {"F16", DType::kFloat16},
      {"BF16", DType::kBFloat16}, {"F32", DType::kFloat32},
      {"F64", DType::kFloat64}};

  auto it = dtype_map.find(v);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return DType::kFloat32;
}

std::string dtype_to_string(DType d) {
  static const std::unordered_map<DType, std::string> dtype_map = {
      {DType::kBool, "BOOL"},     {DType::kUInt8, "U8"},
      {DType::kInt8, "I8"},       {DType::kUInt16, "U16"},
      {DType::kInt16, "I16"},     {DType::kUInt32, "U32"},
      {DType::kInt32, "I32"},     {DType::kUInt64, "U64"},
      {DType::kInt64, "I64"},     {DType::kFloat16, "F16"},
      {DType::kBFloat16, "BF16"}, {DType::kFloat32, "F32"},
      {DType::kFloat64, "F64"}};

  auto it = dtype_map.find(d);
  if (it != dtype_map.end()) {
    return it->second;
  }
  return "F32";
}

size_t get_element_size(DType dtype) {
  size_t element_size = 0;
  switch (dtype) {
  case kBool:
  case kUInt8:
  case kInt8:
    element_size = 1;
    break;
  case kUInt16:
  case kInt16:
  case kFloat16:
  case kBFloat16:
    element_size = 2;
    break;
  case kUInt32:
  case kInt32:
  case kFloat32:
    element_size = 4;
    break;
  case kUInt64:
  case kInt64:
  case kFloat64:
    element_size = 8;
    break;

  default:
    break;
  }
  return element_size;
}

Tensor::Tensor() : dtype_(kFloat32), data_(nullptr) {}

Tensor::~Tensor() = default;

Tensor::Tensor(DType dtype, const std::vector<int64_t> &shape)
    : dtype_(dtype), shape_(shape), data_(nullptr) {
  stride_ = calculate_stride(shape);
}

void Tensor::allocate(Storage *storage) {
  if (data_)
    return;
  auto size = data_size();
  if (storage) {
    data_ = storage->allocate(size);
  } else {
    storage_ = std::make_shared<Storage>(size);
    data_ = storage_->data();
  }
}

Tensor Tensor::from_blob(DType dtype, const std::vector<int64_t> &shape,
                         void *data) {
  Tensor tensor = Tensor(dtype, shape);
  tensor.data_ = reinterpret_cast<uint8_t *>(data);
  return tensor;
}

Tensor Tensor::zeros(DType dtype, const std::vector<int64_t> &shape) {
  Tensor tensor(dtype, shape);
  tensor.allocate();
  memset(tensor.data(), 0, tensor.data_size());
  return tensor;
}

Tensor Tensor::ones(DType dtype, const std::vector<int64_t> &shape) {
  Tensor tensor(dtype, shape);
  tensor.allocate();
  auto num_elements = tensor.numel();
  uint8_t *data = tensor.data();
  // 使用通用分发器和lambda来填充数据
  dispatch_fill_by_dtype(dtype, data, num_elements,
                         [](auto *ptr, size_t count) {
                           using T = std::remove_pointer_t<decltype(ptr)>;

                           if constexpr (std::is_same_v<T, bool>) {
                             std::fill(ptr, ptr + count, true);
                           } else if constexpr (std::is_floating_point_v<T>) {
                             std::fill(ptr, ptr + count, static_cast<T>(1.0));
                           } else if constexpr (std::is_integral_v<T>) {
                             // 整数类型（包括kFloat16/kBFloat16转换为的uint16_t）
                             std::fill(ptr, ptr + count, static_cast<T>(1));
                           }
                         });
  return tensor;
}

Tensor Tensor::randn(DType dtype, const std::vector<int64_t> &shape) {
  // 创建指定类型和形状的空张量
  Tensor tensor(dtype, shape);
  tensor.allocate();
  size_t num_elements = tensor.numel();
  uint8_t *data = tensor.data();

  // 使用通用分发器和lambda来填充随机正态分布数据
  dispatch_fill_by_dtype(
      dtype, data, num_elements, [](auto *ptr, size_t count) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        for (size_t i = 0; i < count; ++i) {
          double val = normal_dist(generator);

          if constexpr (std::is_same_v<T, bool>) {
            ptr[i] = (val > 0.0);
          } else if constexpr (std::is_floating_point_v<T>) {
            ptr[i] = static_cast<T>(val);
          } else if constexpr (std::is_integral_v<T>) {
            // 统一处理所有整数类型
            if constexpr (std::is_same_v<T, int64_t>) {
              ptr[i] = static_cast<T>(std::llround(val));
            } else if constexpr (std::is_same_v<T, uint64_t>) {
              // 对于无符号类型，确保值为非负
              ptr[i] = static_cast<T>(std::llround(std::max(0.0, val)));
            } else if constexpr (std::is_signed_v<T>) {
              ptr[i] = static_cast<T>(std::round(val));
            } else { // 无符号整数类型
              ptr[i] = static_cast<T>(std::round(std::max(0.0, val)));
            }
          }
        }
      });
  return tensor;
}

Tensor Tensor::rand(DType dtype, const std::vector<int64_t> &shape) {
  // 创建指定类型和形状的空张量
  Tensor tensor(dtype, shape);
  tensor.allocate();
  size_t num_elements = tensor.numel();
  uint8_t *data = tensor.data();

  // 使用通用分发器和lambda来填充随机均匀分布数据
  dispatch_fill_by_dtype(
      dtype, data, num_elements, [](auto *ptr, size_t count) {
        using T = std::remove_pointer_t<decltype(ptr)>;

        for (size_t i = 0; i < count; ++i) {
          double val = uniform_real_dis(generator);

          if constexpr (std::is_same_v<T, bool>) {
            ptr[i] = (val > 0.5); // 使用0.5作为阈值更合理
          } else if constexpr (std::is_floating_point_v<T>) {
            ptr[i] = static_cast<T>(val);
          } else if constexpr (std::is_integral_v<T>) {
            // 统一处理所有整数类型
            if constexpr (std::is_same_v<T, int64_t>) {
              ptr[i] = static_cast<T>(std::llround(val));
            } else if constexpr (std::is_same_v<T, uint64_t>) {
              ptr[i] = static_cast<T>(std::llround(std::max(0.0, val)));
            } else if constexpr (std::is_signed_v<T>) {
              ptr[i] = static_cast<T>(std::round(val));
            } else { // 无符号整数类型
              ptr[i] = static_cast<T>(std::round(std::max(0.0, val)));
            }
          }
        }
      });
  return tensor;
}

Tensor Tensor::blank(DType dtype, const std::vector<int64_t> &shape) {
  Tensor tensor(dtype, shape);
  tensor.allocate();
  return tensor;
}


size_t Tensor::numel() const {
  return std::accumulate(std::begin(shape_), std::end(shape_), 1,
                         std::multiplies<>());
}

size_t Tensor::data_size() const { return numel() * get_element_size(dtype_); }

int64_t Tensor::size(int64_t dim) const {
  if (dim < 0) {
    dim += shape_.size();
  }
  return shape_[dim];
}

int64_t Tensor::stride(int64_t dim) const {
  if (dim < 0) {
    dim += shape_.size();
  }
  return stride_[dim];
}

Tensor Tensor::clone() const {
  auto new_tensor = Tensor(dtype_, shape_);
  new_tensor.allocate();
  std::memcpy(new_tensor.data(), data(), data_size());
  return new_tensor;
}

Tensor Tensor::transpose_2d() {
  if (dim() != 2) {
    return *this;
  }
  auto B = size(0);
  auto C = size(1);
  size_t element_size = get_element_size(dtype_);
  size_t total_data_size = data_size();

  std::unique_ptr<uint8_t[]> temp_buffer(new uint8_t[total_data_size]);

  auto src_data = reinterpret_cast<const uint8_t *>(data());
  auto dst_data = temp_buffer.get();

  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < C; ++j) {
      // 计算源（原始张量）和目标（转置后的临时缓冲区）的线性偏移量
      size_t src_linear_offset = (i * C + j) * element_size;
      size_t dst_linear_offset = (j * B + i) * element_size;
      std::memcpy(dst_data + dst_linear_offset, src_data + src_linear_offset,
                  element_size);
    }
  }
  std::memcpy(data(), dst_data, total_data_size);
  std::swap(shape_[0], shape_[1]);
  stride_ = calculate_stride(shape_);
  return *this;
}
} // namespace dense