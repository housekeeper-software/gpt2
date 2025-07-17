#ifndef STORAGE_H_
#define STORAGE_H_

#include <memory>

namespace dense {
class Storage {
public:
  Storage();
  Storage(size_t size);
  ~Storage();
  bool is_valid() const { return capacity_ > 0; }

  size_t capacity() const { return capacity_; }

  uint8_t *data() { return data_.get(); }

  const uint8_t *data() const { return data_.get(); }

  uint8_t *allocate(size_t size);

private:
  std::unique_ptr<uint8_t[]> data_;
  size_t capacity_;
  size_t offset_;
};
} // namespace dense

#endif