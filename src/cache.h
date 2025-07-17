#ifndef CACHE_H_
#define CACHE_H_

#include "storage.h"
#include "tensor.h"
#include <vector>

class DynamicTensor {
public:
  DynamicTensor(size_t max_token);
  ~DynamicTensor() = default;

  const dense::Tensor &get() const { return state_; }
  void update(const dense::Tensor &new_tensor);
  void reset();

private:
  size_t max_token_;
  dense::Tensor state_;
  std::shared_ptr<dense::Storage> storage_;
  std::shared_ptr<dense::Storage> temp_storage_;
};

class CacheLayer {
public:
  CacheLayer(size_t max_token);
  ~CacheLayer() = default;
  const dense::Tensor &key_states() const;
  const dense::Tensor &value_states() const;
  void update(const dense::Tensor &new_key, const dense::Tensor &new_value);
  void reset();

private:
  DynamicTensor key_state_;
  DynamicTensor value_state_;
};

class DynamicCache {
public:
  DynamicCache(size_t n_layers, size_t max_token);
  ~DynamicCache() = default;
  CacheLayer *get(size_t layer_idx);

  int64_t DynamicCache::get_seq_length(size_t layer_idx = 0) const;
  void reset();

private:
  std::vector<CacheLayer> layers_;
};

#endif // CACHE_H_