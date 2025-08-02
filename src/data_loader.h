#ifndef DATA_LOADER_H_
#define DATA_LOADER_H_

#include "tensor.h"
#include <random>
#include <string>

class TikTokenizer;

bool PreProcessData(const std::string &in_file, const std::string &out_file,
                    TikTokenizer *tokenizer);

dense::Tensor LoadPreProcessedData(const std::string &filename);

class TextDataset {
public:
  TextDataset(dense::Tensor data_sequence, int context_length);
  ~TextDataset() = default;
  size_t size() const;
  std::pair<dense::Tensor, dense::Tensor> get(size_t idx) const;

private:
  int context_length_;
  dense::Tensor data_sequence_;
};

class DataLoader {
public:
  DataLoader(std::unique_ptr<TextDataset> dataset, size_t batch_size,
             bool shuffle);
  ~DataLoader() = default;
  size_t bacth_size() const { return batch_size_; }
  size_t size() const {
    return (dataset_->size() + batch_size_ - 1) / batch_size_;
  }

  class Iterator {
  private:
    DataLoader *parent_;
    size_t current_batch_start_idx_ = 0;

    dense::Tensor stack(const std::vector<dense::Tensor> &vec) const {
      int64_t B = vec.size();
      dense::Tensor output =
          dense::Tensor::zeros(vec[0].dtype(), {B, vec[0].size(0)});
      auto ptr = output.data();
      for (const auto &i : vec) {
        memcpy(ptr, i.data(), i.data_size());
        ptr += i.data_size();
      }
      return output;
    }

  public:
    Iterator(DataLoader *parent, size_t start_idx)
        : parent_(parent), current_batch_start_idx_(start_idx) {}

    // 前置递增运算符
    Iterator &operator++() {
      current_batch_start_idx_ += parent_->batch_size_;
      return *this;
    }

    // 不等式运算符
    bool operator!=(const Iterator &other) const {
      return current_batch_start_idx_ != other.current_batch_start_idx_;
    }

    // 解引用运算符，返回当前批次数据
    std::pair<dense::Tensor, dense::Tensor> operator*() {
      std::vector<dense::Tensor> x_batch_vec;
      std::vector<dense::Tensor> y_batch_vec;

      size_t batch_end_idx =
          std::min(current_batch_start_idx_ + parent_->batch_size_,
                   parent_->indices_.size());

      for (size_t i = current_batch_start_idx_; i < batch_end_idx; ++i) {
        // 使用打乱后的索引获取样本
        auto example = parent_->dataset_->get(parent_->indices_[i]);
        x_batch_vec.push_back(example.first);
        y_batch_vec.push_back(example.second);
      }

      // 将单个样本张量堆叠成批次张量
      dense::Tensor x_batch = stack(x_batch_vec);
      dense::Tensor y_batch = stack(y_batch_vec);

      return {x_batch, y_batch};
    }
  };

  // Begin 迭代器
  Iterator begin() {
    // 如果需要洗牌，每个 epoch 开始时重新洗牌
    if (shuffle_) {
      std::shuffle(indices_.begin(), indices_.end(), rng_);
    }
    return Iterator(this, 0);
  }

  // End 迭代器
  Iterator end() {
    // 计算最后一个批次的起始索引，可能不是批次大小的整数倍
    size_t dataset_size = dataset_->size();
    size_t num_batches = (dataset_size + batch_size_ - 1) / batch_size_;
    size_t end_idx =
        num_batches * batch_size_; // 最后一个批次结束后，下一个批次的开始索引
    return Iterator(this, end_idx);
  }

private:
  std::unique_ptr<TextDataset> dataset_; // 使用 shared_ptr 管理数据集生命周期
  size_t batch_size_;
  bool shuffle_;
  std::vector<size_t> indices_; // 存储数据集的索引，用于洗牌和迭代
  std::mt19937 rng_;            // 随机数生成器用于洗牌
};

#endif // DATA_LOADER_H_