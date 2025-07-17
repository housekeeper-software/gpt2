#ifndef TENSOR_H_
#define TENSOR_H_

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace dense {

class Storage;

enum DType {
  kBool,
  kUInt8,
  kInt8,
  kUInt16,
  kInt16,
  kUInt32,
  kInt32,
  kUInt64,
  kInt64,
  kFloat16,
  kBFloat16,
  kFloat32,
  kFloat64,
};

DType dtype_from_string(const std::string &v);

std::string dtype_to_string(DType d);

size_t get_element_size(DType dtype);

class Tensor {
public:
  Tensor();
  Tensor(DType dtype, const std::vector<int64_t> &shape);
  ~Tensor();

  Tensor(const Tensor &other) = default;
  Tensor(Tensor &&other) noexcept = default;
  Tensor &operator=(const Tensor &other) = default;

  static Tensor from_blob(DType dtype, const std::vector<int64_t> &shape,
                          void *data);

  static Tensor zeros(DType dtype, const std::vector<int64_t> &shape);
  static Tensor ones(DType dtype, const std::vector<int64_t> &shape);
  static Tensor randn(DType dtype, const std::vector<int64_t> &shape);
  static Tensor rand(DType dtype, const std::vector<int64_t> &shape);
  static Tensor blank(DType dtype, const std::vector<int64_t> &shape);

  uint8_t *data() { return data_; }

  const uint8_t *data() const { return data_; }

  size_t data_size() const;

  size_t numel() const;

  DType dtype() const { return dtype_; }

  bool is_defined() const { return !shape_.empty(); }

  bool empty() const { return data_ == nullptr; }

  int64_t dim() const { return shape_.size(); }

  int64_t size(int64_t dim) const;

  int64_t stride(int64_t dim) const;

  std::vector<int64_t> shape() const { return shape_; }

  Tensor transpose_2d();

  Tensor clone() const;

  std::string to_string() const;

  void allocate(Storage *storage = nullptr);

private:
  DType dtype_;
  std::vector<int64_t> shape_;
  std::vector<size_t> stride_;
  uint8_t *data_;
  std::shared_ptr<Storage> storage_;
};

} // namespace dense

#endif