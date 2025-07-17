#ifndef SAMPLING_H_
#define SAMPLING_H_

#include "tensor.h"
#include <random>
#include <unordered_map>
#include <vector>

// copy from llama.cpp

typedef int32_t llama_token;

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

struct llama_token_data {
  llama_token id;
  float logit;
  float p;
  llama_token_data() : id(0), logit(0.0f), p(0.0f) {}
  llama_token_data(llama_token id, float logit, float p)
      : id(id), logit(logit), p(p) {}
};

struct llama_token_data_array {
  llama_token_data_array(llama_token_data *data, size_t size)
      : data(data), size(size), selected(-1), sorted(false) {}

  llama_token_data *data;
  size_t size;
  int64_t selected;
  bool sorted;
};

template <typename T> struct ring_buffer {
  ring_buffer(size_t cap) : capacity(cap), data(cap) {}

  T &front() {
    if (sz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[first];
  }

  const T &front() const {
    if (sz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[first];
  }

  T &back() {
    if (sz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[pos];
  }

  const T &back() const {
    if (sz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    return data[pos];
  }

  void push_back(const T &value) {
    if (capacity == 0) {
      throw std::runtime_error("ring buffer: capacity is zero");
    }

    if (sz == capacity) {
      // advance the start when buffer is full
      first = (first + 1) % capacity;
    } else {
      sz++;
    }
    data[pos] = value;
    pos = (pos + 1) % capacity;
  }

  T pop_front() {
    if (sz == 0) {
      throw std::runtime_error("ring buffer is empty");
    }
    T value = data[first];
    first = (first + 1) % capacity;
    sz--;
    return value;
  }

  const T &rat(size_t i) const {
    if (i >= sz) {
      throw std::runtime_error("ring buffer: index out of bounds");
    }
    return data[(first + sz - i - 1) % capacity];
  }

  std::vector<T> to_vector() const {
    std::vector<T> result;
    result.reserve(sz);
    for (size_t i = 0; i < sz; i++) {
      result.push_back(data[(first + i) % capacity]);
    }
    return result;
  }

  void clear() {
    // here only reset the status of the buffer
    sz = 0;
    first = 0;
    pos = 0;
  }

  bool empty() const { return sz == 0; }

  size_t size() const { return sz; }

  size_t capacity = 0;
  size_t sz = 0;
  size_t first = 0;
  size_t pos = 0;

  std::vector<T> data;
};

class Sampling {
public:
  Sampling() = default;
  virtual ~Sampling() = default;

  virtual void apply(llama_token_data_array *cur_p) = 0;
  virtual void accept(llama_token token) {}
  virtual void reset() {}

  virtual std::string name() const { return "sampling"; }
};

/*
Greedy (贪心采样)
这是最简单的策略。它总是在所有候选 token 中选择 logit
值最高（也就是概率最高）的那一个
*/
class GreedySampling : public Sampling {
public:
  GreedySampling() = default;
  ~GreedySampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "greedy"; }
};

/*
Distribution / Multinomial (分布/多项式采样)
这是最经典的随机采样方法。它将所有候选 token 的 logit 值通过 Softmax
函数转换为一个概率分布，然后根据每个 token 的概率大小进行加权随机抽样
*/
class DistSampling : public Sampling {
public:
  DistSampling(uint32_t seed);
  ~DistSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;

  std::string name() const override { return "dist"; }

private:
  uint32_t seed_;
  uint32_t seed_cur_;
  std::mt19937 rng_;
};

/*
Top-K 采样
将候选 token 限制在 logit 值最高的 K 个 以内。例如，如果
k=50，则只会在概率最高的 50 个 token
中进行采样，其余的全部丢弃。这可以有效防止选中一些非常奇怪但概率不为零的词
*/
class TopKSampling : public Sampling {
public:
  TopKSampling(int32_t k) : k_(k) {}
  ~TopKSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "top-k"; }

private:
  int32_t k_;
};

/*
Top-P (Nucleus) 采样
选择一个概率核心（Nucleus）。它会按概率从高到低累加，直到累加的概率总和超过设定的阈值
p。然后，采样过程只在这个核心 token 集合中进行。与 Top-K 不同，Top-P
的候选池大小是动态的
*/
class TopPSampling : public Sampling {
public:
  TopPSampling(float p, size_t min_keep) : p_(p), min_keep_(min_keep) {}
  ~TopPSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "top-p"; }

private:
  float p_;
  size_t min_keep_;
};

/*
Min-P 采样
它会保留所有概率不低于 最高概率 * p 的 token。这确保了候选 token 与最可能的
token 相比不会太差
*/
class MinPSampling : public Sampling {
public:
  MinPSampling(float p, size_t min_keep) : p_(p), min_keep_(min_keep) {}
  ~MinPSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "min-p"; }

private:
  float p_;
  size_t min_keep_;
};

/*
Typical (典型) 采样
基于论文 "Locally Typical
Sampling"。它旨在过滤掉那些虽然概率不低，但信息量与上下文的平均信息量（熵）差异过大的“非典型”词元
逻辑比较复杂。它计算概率分布的熵，然后计算每个 token 的 -log(p)
与熵的差的绝对值，并根据这个新分数对 token 排序，最后执行类似于 Top-P
的操作来筛选
*/
class TypicalSampling : public Sampling {
public:
  TypicalSampling(float p, size_t min_keep) : p_(p), min_keep_(min_keep) {}
  ~TypicalSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "typical"; }

private:
  float p_;
  size_t min_keep_;
};

/*
Temperature (温度采样)
调整概率分布的“尖锐”程度。

temp > 1.0：使概率分布更平滑，低概率的 token
更容易被选中，增加生成文本的随机性和创造性。

temp < 1.0：使概率分布更尖锐，高概率的 token 更容易被选中，使生成更稳定和保守。

temp <= 0：等同于贪心采样
*/
class TemperatureSampling : public Sampling {
public:
  TemperatureSampling(float t) : temp_(t) {}
  ~TemperatureSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "temp"; }

private:
  float temp_;
};

/*
Dynamic Temperature (动态温度)
将候选 token 限制在 logit 值最高的 K 个 以内。例如，如果
k=50，则只会在概率最高的 50 个 token
中进行采样，其余的全部丢弃。这可以有效防止选中一些非常奇怪但概率不为零的词
会对候选 token 按 logit 排序，然后将候选池的大小 cur_p->size 截断为 k
*/
class TemperatureExSampling : public Sampling {
public:
  TemperatureExSampling(float t, float delta, float exponent)
      : temp_(t), delta_(delta), exponent_(exponent) {}
  ~TemperatureExSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "temp-ext"; }

private:
  float temp_;
  float delta_;
  float exponent_;
};

/*
XTC (Exclude The Top Choice) 采样
一种实验性的采样器。它会以一定的概率 (p)
触发，触发后会从候选池中移除概率高于某个阈值 (t) 的
token。这是一种反向操作，旨在强制模型跳出最显而易见的选项，探索更多可能性
通过一个随机数决定是否激活。如果激活，它会移除 data 数组开头部分（概率最高的）的
token
*/
class XTCSampling : public Sampling {
public:
  XTCSampling(float p, float threshold, size_t min_keep,
              uint32_t seed = LLAMA_DEFAULT_SEED);
  ~XTCSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;

  std::string name() const override { return "xtc"; }

private:
  float probability_;
  float threshold_;
  size_t min_keep_;
  uint32_t seed_;
  uint32_t seed_cur_;
  std::mt19937 rng_;
};

/*
Top-n-sigma 采样
一种基于统计的过滤方法。它计算所有 logits 的均值和标准差，并保留所有 logit
值不低于 max_logit - n * std_dev 的 token。它旨在移除统计上的异常值
*/
class TopNSigmaSampling : public Sampling {
public:
  TopNSigmaSampling(float n) : n_(n) {}
  ~TopNSigmaSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;

  std::string name() const override { return "top-n-sigma"; }

private:
  float n_;
};

/*
Mirostat (v1 & v2)
一种基于反馈的采样算法。它不直接控制概率，而是试图将生成文本的“惊奇度”（Perplexity）维持在一个恒定的目标水平
tau。这可以防止文本变得过于平淡或过于混乱 v1: 比较复杂，通过估算 s_hat
来动态计算一个 k 值，然后执行 Top-K v2: 更直接，它会截断所有“惊奇度” (-log2(p))
高于其内部状态 mu 的 token。每次采样后，它会根据实际采样到的 token
的惊奇度与目标 tau 之间的误差来更新 mu
*/
class MirostatSampling : public Sampling {
public:
  MirostatSampling(int32_t n_vocab, uint32_t seed, float tau, float eta,
                   int32_t m);
  ~MirostatSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;

  std::string name() const override { return "mirostat"; }

private:
  int32_t n_vocab_;
  uint32_t seed_;
  uint32_t seed_cur_;
  float tau_;
  float eta_;
  int32_t m_;
  float mu_;
  std::mt19937 rng_;
};

class MirostatV2Sampling : public Sampling {
public:
  MirostatV2Sampling(uint32_t seed, float tau, float eta);
  ~MirostatV2Sampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;
  std::string name() const override { return "mirostat-v2"; }

private:
  uint32_t seed_;
  uint32_t seed_cur_;
  float tau_;
  float eta_;
  float mu_;
  std::mt19937 rng_;
};

/*
Standard Penalties (标准惩罚)
对最近 penalty_last_n 个已生成的 token 施加惩罚。
penalty_repeat: 对所有重复的 token 施加除法惩罚（logit /= penalty_repeat）。
penalty_freq: 根据 token 的重复频率施加减法惩罚，重复次数越多，惩罚越重。
penalty_present: 只要 token 在近期出现过，就施加一个固定的减法惩罚
维护了一个近期 token 的频率计数器，并在采样前应用上述惩罚逻辑
*/
class PenaltiesSampling : public Sampling {
public:
  PenaltiesSampling(
      int32_t penalty_last_n, // last n tokens to penalize (0 = disable penalty,
                              // -1 = context size)
      float penalty_repeat,   // 1.0 = disabled
      float penalty_freq,     // 0.0 = disabled
      float penalty_present); // 0.0 = disabled
  ~PenaltiesSampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;
  void accept(llama_token token) override;

  std::string name() const override { return "penalties"; }

private:
  int32_t penalty_last_n_;
  float penalty_repeat_;
  float penalty_freq_;
  float penalty_present_;

  ring_buffer<llama_token> prev_;
  // a frequency map to count token occurrences
  std::unordered_map<llama_token, int> token_count_;
};

/*
DRY (Don't Repeat Yourself) 采样
一种更高级、更强力的重复惩罚机制。它专门针对多词元序列 (multi-token sequences)
的重复。它使用 Z-algorithm
等算法来高效地检测上下文中的重复长序列，并对将要延续这个重复序列的 token
施加巨大的、呈指数增长的惩罚 包含了序列匹配、Z-algorithm 应用、以及基于
pow(dry_base, exponent) 的指数惩罚计算。它比标准惩罚更能有效阻止长句子的重
*/
class DrySampling : public Sampling {
public:
  DrySampling(
      int32_t context_size, float dry_multiplier, float dry_base,
      int32_t dry_allowed_length, int32_t dry_penalty_last_n,
      const std::unordered_multimap<llama_token, std::vector<llama_token>>
          &processed_breakers);
  ~DrySampling() override = default;

  void apply(llama_token_data_array *cur_p) override;
  void reset() override;
  void accept(llama_token token) override;

  std::string name() const override { return "dry"; }

private:
  int32_t total_context_size_;

  const float dry_multiplier_;
  const float dry_base_;
  const int32_t dry_allowed_length_;
  const int32_t dry_penalty_last_n_;

  std::unordered_multimap<llama_token, std::vector<llama_token>>
      dry_processed_breakers_;
  std::vector<int> dry_repeat_count_;
  std::unordered_map<llama_token, int> dry_max_token_repeat_;
  ring_buffer<llama_token> last_tokens_;
};

class SamplingChain {
public:
  SamplingChain() = default;
  ~SamplingChain() = default;
  void add(std::unique_ptr<Sampling> sampling);
  llama_token sample(llama_token_data_array *cur_p);
  void reset();

private:
  void llama_sampler_accept(llama_token token);
  void llama_sampler_apply(llama_token_data_array *cur_p);
  void llama_sampler_reset();
  std::vector<std::unique_ptr<Sampling>> samples_;
};

#endif // SAMPLING_H_