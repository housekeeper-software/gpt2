#include "storage.h"

namespace dense {
namespace {

const size_t kTensorMemAlign = 16;
}

Storage::Storage() : capacity_(0), offset_(0) {}
Storage::Storage(size_t size) : capacity_(size), offset_(0) {
  if (capacity_ > 0) {
    auto capacity = (capacity_ + kTensorMemAlign - 1) & ~(kTensorMemAlign - 1);
    data_.reset(new uint8_t[capacity]);
  }
}

Storage::~Storage() = default;

uint8_t *Storage::allocate(size_t size) {
  if (size + offset_ > capacity_)
    return nullptr;
  auto start = offset_;
  offset_ += size;
  return data() + start;
}

} // namespace dense