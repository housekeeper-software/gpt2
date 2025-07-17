#include "tensor.h"
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <vector> // 确保包含 vector

namespace dense {

namespace {
// 格式化单个元素的值
template <typename T>
std::string format_element(const uint8_t *data, size_t index) {
  // index 此时是元素索引，直接使用
  T value = reinterpret_cast<const T *>(data)[index];
  std::ostringstream oss;

  if constexpr (std::is_same_v<T, bool>) {
    oss << (value ? "true" : "false");
  } else if constexpr (std::is_floating_point_v<T>) {
    // 改为科学计数法显示，保留6位有效数字
    oss << std::scientific << std::setprecision(6) << value;
  } else {
    oss << std::to_string(value);
  }

  return oss.str();
}

// 根据数据类型格式化元素
// index 参数现在代表的是线性数组中的“元素索引”
std::string format_element_by_dtype(DType dtype, const uint8_t *data,
                                    size_t index) {
  switch (dtype) {
  case kBool:
    return format_element<bool>(data, index);
  case kUInt8:
    return format_element<uint8_t>(data, index);
  case kInt8:
    return format_element<int8_t>(data, index);
  case kUInt16:
    return format_element<uint16_t>(data, index);
  case kInt16:
    return format_element<int16_t>(data, index);
  case kUInt32:
    return format_element<uint32_t>(data, index);
  case kInt32:
    return format_element<int32_t>(data, index);
  case kUInt64:
    return format_element<uint64_t>(data, index);
  case kInt64:
    return format_element<int64_t>(data, index);
  case kFloat32:
    return format_element<float>(data, index);
  case kFloat64:
    return format_element<double>(data, index);
  case kFloat16:
  case kBFloat16:
    // 对于 float16 和 bfloat16，暂时当作 uint16 处理
    return format_element<uint16_t>(data, index);
  default:
    return "?";
  }
}

// 用于控制打印的常量
const int64_t MAX_PRINT_ELEMENTS = 6;    // 每个维度最多显示的元素数
const int64_t EDGE_ITEMS = 3;            // 省略时前后各显示的元素数 (调整为3，避免与MAX_PRINT_ELEMENTS冲突过大)
const int64_t MAX_TOTAL_ELEMENTS = 1000; // 总元素数超过此值时会省略

// 检查是否需要省略
bool should_summarize(const std::vector<int64_t> &shape) {
  int64_t total_elements = 1;
  for (int64_t dim : shape) {
    total_elements *= dim;
    if (total_elements > MAX_TOTAL_ELEMENTS) {
      return true;
    }
  }
  return false;
}

// 计算多维索引对应的线性索引 (以元素个数为单位的偏移量)
size_t calculate_linear_index(const std::vector<int64_t> &indices,
                              const std::vector<size_t> &stride) {
  size_t linear_index = 0;
  for (size_t i = 0; i < indices.size(); ++i) {
    linear_index += indices[i] * stride[i];
  }
  return linear_index;
}

// 递归打印多维张量（支持省略）
void print_tensor_recursive(std::ostringstream &oss,
                            const std::vector<int64_t> &shape,
                            const std::vector<size_t> &stride, // 这个 stride 是元素数量的步长
                            const uint8_t *data, DType dtype,
                            std::vector<int64_t> &indices, int current_dim,
                            int indent_level, bool summarize) {
  if (current_dim == shape.size()) {
    // 到达最后一个维度，打印元素
    size_t element_index = calculate_linear_index(indices, stride); // 这已经是以元素个数为单位的偏移量了
    oss << format_element_by_dtype(dtype, data, element_index);
    return;
  }

  // 添加缩进
  std::string current_indent(indent_level * 2, ' ');
  std::string next_level_indent((indent_level + 1) * 2, ' ');

  oss << "[";

  int64_t dim_size = shape[current_dim];
  bool need_ellipsis = summarize && dim_size > MAX_PRINT_ELEMENTS;

  if (need_ellipsis) {
    // 打印前面的元素
    for (int64_t i = 0; i < EDGE_ITEMS; ++i) {
      if (i > 0) {
        oss << ",";
        if (current_dim == shape.size() - 1) { // 最后一个维度，元素之间用空格
          oss << " ";
        } else { // 非最后一个维度，换行并增加缩进
          oss << "\n" << next_level_indent;
        }
      }
      indices[current_dim] = i;
      print_tensor_recursive(oss, shape, stride, data, dtype, indices,
                             current_dim + 1, indent_level + 1, summarize);
    }

    // 添加省略符号
    oss << ", ..."; // 简化省略号的逗号处理
    
    // 打印后面的元素
    for (int64_t i = dim_size - EDGE_ITEMS; i < dim_size; ++i) {
      oss << ",";
      if (current_dim == shape.size() - 1) {
        oss << " ";
      } else {
        oss << "\n" << next_level_indent;
      }
      indices[current_dim] = i;
      print_tensor_recursive(oss, shape, stride, data, dtype, indices,
                             current_dim + 1, indent_level + 1, summarize);
    }
  } else {
    // 打印所有元素
    for (int64_t i = 0; i < dim_size; ++i) {
      if (i > 0) {
        oss << ",";
        if (current_dim == shape.size() - 1) {
          oss << " ";
        } else {
          oss << "\n" << next_level_indent;
        }
      }
      indices[current_dim] = i;
      print_tensor_recursive(oss, shape, stride, data, dtype, indices,
                             current_dim + 1, indent_level + 1, summarize);
    }
  }

  // 结束当前维度，如果不是最外层，则退格
  oss << "]";
  if (current_dim > 0 && shape.size() > 1 && dim_size > 0 && (need_ellipsis || dim_size > 1)) {
      if (current_dim == shape.size() -1 && !need_ellipsis && dim_size <= 1) {
          // 最后一维，元素不多时，且没有省略号，不加额外换行
      } else {
          oss << "\n" << current_indent;
      }
  }
}
} // namespace

std::string Tensor::to_string() const {
  std::ostringstream oss;

  // 处理空张量
  if (empty()) {
    oss << "tensor([], dtype=" << dtype_to_string(dtype_) << ")";
    return oss.str();
  }

  // 处理标量（0维张量）
  if (shape_.empty()) {
    size_t element_index = 0;
    oss << "tensor(" << format_element_by_dtype(dtype_, data(), element_index)
        << ", dtype=" << dtype_to_string(dtype_) << ")";
    return oss.str();
  }

  // 检查是否需要省略
  bool summarize = should_summarize(shape_);

  // 处理多维张量
  oss << "tensor(";

  std::vector<int64_t> indices(shape_.size(), 0);
  print_tensor_recursive(oss, shape_, stride_, data(), dtype_, indices, 0, 0,
                         summarize);

  oss << ", dtype=" << dtype_to_string(dtype_);

  // 添加形状信息
  oss << ", shape=[";
  for (size_t i = 0; i < shape_.size(); ++i) {
    if (i > 0)
      oss << ", ";
    oss << shape_[i];
  }
  oss << "]";

  if (summarize) {
    oss << ", summarized=true";
  }

  oss << ")";

  return oss.str();
}
} // namespace dense