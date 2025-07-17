#include "cache.h"
#include <stdexcept>

DynamicTensor::DynamicTensor(size_t max_token) : max_token_(max_token) {}

void DynamicTensor::reset() { state_ = dense::Tensor(); }

void DynamicTensor::update(const dense::Tensor &new_tensor) {
  auto element_size = dense::get_element_size(new_tensor.dtype());
  auto B = new_tensor.size(0);
  auto C = new_tensor.size(2);

  if (!state_.is_defined()) {
    storage_ =
        std::make_shared<dense::Storage>(B * max_token_ * C * element_size);
    state_ = dense::Tensor::from_blob(new_tensor.dtype(), new_tensor.shape(),
                                      storage_->data());
    std::memcpy(state_.data(), new_tensor.data(), new_tensor.data_size());
  } else {
    if (B == 1) {
      //推理的时候，一般批次都是1，这里就是快捷方式，直接将新的张量数据复制到尾部即可
      auto NEW_T = state_.size(1) + new_tensor.size(1);
      auto new_state = dense::Tensor::from_blob(state_.dtype(), {B, NEW_T, C},
                                                storage_->data());
      std::memcpy(new_state.data() + state_.data_size(), new_tensor.data(),
                  new_tensor.data_size());
      state_ = std::move(new_state);

    } else {
      if (!temp_storage_) {
        //我们要借助缓冲张量来完成复制，因为相同批次是连续的，所以会分批次复制
        temp_storage_ =
            std::make_shared<dense::Storage>(B * max_token_ * C * element_size);
      }
      auto NEW_T = state_.size(1) + new_tensor.size(1);
      auto new_state = dense::Tensor::from_blob(state_.dtype(), {B, NEW_T, C},
                                                temp_storage_->data());

      auto old_batch_size = state_.size(1) * C * element_size;
      auto new_batch_size = new_tensor.size(1) * C * element_size;
      auto result_batch_size = NEW_T * C * element_size;

      for (size_t b_idx = 0; b_idx < B; ++b_idx) {
        auto old_src = state_.data() + b_idx * old_batch_size;
        auto new_src = new_tensor.data() + b_idx * new_batch_size;
        auto dest_base = new_state.data() + b_idx * result_batch_size;
        std::memcpy(dest_base, old_src, old_batch_size);
        std::memcpy(dest_base + old_batch_size, new_src, new_batch_size);
      }
      std::swap(storage_, temp_storage_);
      state_ = dense::Tensor::from_blob(new_state.dtype(), new_state.shape(),
                                        storage_->data());
    }
  }
}

CacheLayer::CacheLayer(size_t max_token)
    : key_state_(max_token), value_state_(max_token) {}

const dense::Tensor &CacheLayer::key_states() const { return key_state_.get(); }
const dense::Tensor &CacheLayer::value_states() const {
  return value_state_.get();
}

void CacheLayer::reset() {
  key_state_.reset();
  value_state_.reset();
}

void CacheLayer::update(const dense::Tensor &new_key,
                        const dense::Tensor &new_value) {
  key_state_.update(new_key);
  value_state_.update(new_value);
}

DynamicCache::DynamicCache(size_t n_layers, size_t max_token) {
  layers_.reserve(n_layers);
  for (size_t i = 0; i < n_layers; ++i) {
    layers_.emplace_back(CacheLayer(max_token)); // 默认构造空的 CacheLayer。
  }
}

CacheLayer *DynamicCache::get(size_t layer_idx) {
  if (layer_idx < layers_.size()) {
    return &layers_[layer_idx]; // 返回该层对应的 CacheLayer 引用。
  } else {
    throw std::out_of_range("缓存中只有 " + std::to_string(layers_.size()) +
                            " 层，但尝试访问索引为 " +
                            std::to_string(layer_idx) + " 的层。");
  }
}

int64_t DynamicCache::get_seq_length(size_t layer_idx) const {
  if (layers_[layer_idx].key_states().is_defined()) {
    return layers_[layer_idx].key_states().size(-2);
  }
  return 0;
}

void DynamicCache::reset() {
  for (auto &i : layers_) {
    i.reset();
  }
}