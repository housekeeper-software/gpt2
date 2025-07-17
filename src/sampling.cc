#include "sampling.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace {
// Softmax
// 函数将任意实数（logits）转换为0到1之间的概率分布，并且所有概率的和为1。
void llama_sampler_softmax_impl(llama_token_data_array *cur_p) {
  assert(cur_p->size > 0);

  // 如果数据未排序，则按 logit（模型输出的原始分数）降序排序。
  // 许多采样策略都依赖于已排序的 logits，以便快速找到高概率的 token。
  if (!cur_p->sorted) {
    std::sort(cur_p->data, cur_p->data + cur_p->size,
              [](const llama_token_data &a, const llama_token_data &b) {
                // 按照 logit 降序排序，即 logit 值越大越靠前。
                return a.logit > b.logit;
              });
    cur_p->sorted = true; // 标记数组已排序。
  }

  // 找到最大的 logit 值，用于数值稳定性。
  // 在 Softmax 计算中，减去最大 logit 可以防止 expf(logit) 过大导致浮点数溢出。
  // 这里已经排序了，第一个就是最大值
  float max_l = cur_p->data[0].logit;
  float cum_sum = 0.0f;

  for (size_t i = 0; i < cur_p->size; ++i) {
    float p = expf(cur_p->data[i].logit - max_l);
    cur_p->data[i].p = p;
    cum_sum += p;
  }

  for (size_t i = 0; i < cur_p->size; ++i) {
    cur_p->data[i].p /= cum_sum;
  }
}

// 这是一个辅助函数，用于应用温度（Temperature）对 logits 进行调整。
// 温度参数可以控制生成文本的随机性：
// - temp > 1.0 使分布更平坦，增加随机性。
// - temp < 1.0 使分布更尖锐，减少随机性。
// - temp == 0.0 等同于贪心采样。
void llama_sampler_temp_impl(llama_token_data_array *cur_p, float temp) {
  if (temp <= 0.0f) {
    // 如果温度小于等于0，执行贪心选择。
    // 找到 logit 最高的 token，并将其余 token 的 logit 设置为负无穷大。
    // 这实际上是强制只选择最高 logit 的 token。
    size_t max_i = 0;
    float max_l = cur_p->data[0].logit;

    for (size_t i = 1; i < cur_p->size; ++i) {
      if (cur_p->data[i].logit > max_l) {
        cur_p->data[max_i].logit = -INFINITY;
        max_i = i;
        max_l = cur_p->data[i].logit;
      } else {
        cur_p->data[i].logit = -INFINITY;
      }
    }
    return;
  }

  // 如果温度大于0，则将所有 logit 除以温度。
  // 这会改变概率分布的形状：温度越高，logits 之间的差异越小，Softmax
  // 后概率分布越平坦；反之越尖锐。
  for (size_t i = 0; i < cur_p->size; ++i) {
    // 将每个 logit 除以温度。
    cur_p->data[i].logit /= temp;
  }
}

uint32_t get_rng_seed(uint32_t seed) {
  if (seed == LLAMA_DEFAULT_SEED) {
    // 如果传入的种子是默认值 (LLAMA_DEFAULT_SEED)，表示需要一个真随机种子。
    // 检查 std::random_device 是否是真随机数生成器。
    // 某些系统上 std::random_device 可能不是真随机（entropy() == 0）。
    static bool is_rd_prng = std::random_device().entropy() == 0;
    if (is_rd_prng) {
      // 如果不是真随机，则使用系统时钟作为种子。
      return (uint32_t)std::chrono::system_clock::now()
          .time_since_epoch()
          .count();
    }
    // 使用 std::random_device 获取一个高质量的随机种子（如果可用）。
    std::random_device rd;
    return rd();
  }
  return seed;
}

// 执行离散分布采样（多项式采样）。
// 这是根据 token 的概率 p 进行加权随机选择的核心函数。
int llama_sample_dist(llama_token_data_array *cur_p, std::mt19937 &rng) {
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

  // 定义一个迭代器，用于 std::discrete_distribution。
  // std::discrete_distribution 需要一个迭代器范围来获取每个元素的概率。
  struct probs_iterator {
    typedef std::input_iterator_tag iterator_category;
    typedef float value_type;
    typedef float *pointer;
    typedef float &reference;
    typedef ptrdiff_t difference_type;

    const llama_token_data *data;

    bool operator==(const probs_iterator &other) const {
      return data == other.data;
    }
    bool operator!=(const probs_iterator &other) const {
      return data != other.data;
    }
    const float &operator*() const { return data->p; }
    probs_iterator &operator++() {
      ++data;
      return *this;
    }
    probs_iterator operator++(int) {
      probs_iterator tmp = *this;
      ++data;
      return tmp;
    }
  };

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

  // 创建一个 std::discrete_distribution
  // 对象，它根据给定的概率分布生成随机整数。 范围从 cur_p->data 到 cur_p->data
  // + cur_p->size，概率由 probs_iterator 提供。
  std::discrete_distribution<int> dist(
      probs_iterator{cur_p->data}, probs_iterator{cur_p->data + cur_p->size});

  // 使用传入的随机数引擎 rng 生成一个随机索引，该索引的概率与 token 的 p
  // 值成正比。
  return dist(rng);
}

// 辅助函数，用于实现 Top-K 采样。
// Top-K 采样只保留 logit 值最高的 K 个 token，并丢弃其余的。
void llama_sampler_top_k_impl(llama_token_data_array *cur_p, int32_t k) {
  // 如果 k 小于等于 0，不执行任何操作。
  if (k <= 0) {
    return;
  }

  // 确保 k 不超过当前 token 数组的大小。
  k = std::min(k, (int)cur_p->size);

  // 如果数据未排序，则进行排序。
  if (!cur_p->sorted) {
    // 定义一个比较函数，按 logit 降序排序。
    auto comp = [](const llama_token_data &a, const llama_token_data &b) {
      return a.logit > b.logit;
    };
    // 根据 k 的大小选择不同的排序策略以优化性能。
    if (k <= 128) {
      // 如果 k 较小，使用 std::partial_sort。
      // partial_sort 只保证前 k 个元素是已排序的，并且是整个数组中最小的 k
      // 个（这里是最大的 k 个因为是降序）。
      std::partial_sort(cur_p->data, cur_p->data + k, cur_p->data + cur_p->size,
                        comp);
    } else {
      // 如果 k 较大，使用桶排序（Bucket Sort）的优化，可能比完全排序更快。
      // 这个优化是为了在 k 很大的情况下避免完全排序整个数组，从而提高效率。
      constexpr int nbuckets = 128;        // 桶的数量。
      constexpr float bucket_low = -10.0f; // logit 值的下限。
      constexpr float bucket_high = 10.0f; // logit 值的上限。
      constexpr float bucket_scale =
          nbuckets /
          (bucket_high - bucket_low); // 缩放因子，将 logit 值映射到桶索引。
      constexpr float bucket_inter =
          -bucket_low * bucket_scale; // 截距，用于映射。

      // 存储每个 token 对应的桶索引。
      std::vector<int> bucket_idx(cur_p->size);
      // 存储每个桶中的 token 数量（直方图）。
      std::vector<int> histo(nbuckets, 0);

      // 将每个 token 分配到对应的桶，并统计每个桶的 token 数量。
      for (int i = 0; i < (int)cur_p->size; ++i) {
        const float val = cur_p->data[i].logit;
        // 计算桶索引：(val - bucket_low) / (bucket_high - bucket_low) *
        // nbuckets
        // 简化为 bucket_scale * val + bucket_inter
        int ib = int(bucket_scale * val +
                     bucket_inter); // nbuckets * (val - bucket_low) /
                                    // (bucket_high - bucket_low);
                                    // 确保索引在有效范围内 [0, nbuckets-1]。
        ib = std::max(0, std::min(nbuckets - 1, ib));
        // 记录 token 的桶索引。
        bucket_idx[i] = ib;
        // 增加对应桶的计数。
        ++histo[ib];
      }

      // 从最高 logit 的桶（即最大索引的桶）开始累加 token
      // 数量，直到累加数量达到 k。
      // 确定需要保留哪些桶。
      int nhave = 0;         // 已累加的 token 数量。
      int ib = nbuckets - 1; // 从最高索引的桶开始。
      for (; ib >= 0; --ib) {
        nhave += histo[ib]; // 累加当前桶的 token 数量。
        if (nhave >= k) {
          // 如果累加数量达到或超过 k。
          // 停止，我们找到了包含第 k 个元素的桶。
          break;
        }
      }

      // 创建一个临时向量来存储选定的 token。
      std::vector<llama_token_data> tmp_tokens(nhave);
      // 指向临时向量的指针。
      auto *ptr = tmp_tokens.data();
      // 存储每个桶在 tmp_tokens 中的起始位置。
      std::vector<llama_token_data *> bucket_ptrs;
      // 预留空间。
      bucket_ptrs.reserve(nbuckets - ib);
      for (int j = nbuckets - 1; j >= ib; --j) {
        // 将当前 ptr 压入，作为该桶的起始地址。
        bucket_ptrs.push_back(ptr);
        // ptr 向后移动，为下一个桶准备空间。
        ptr += histo[j];
      }

      // 将原始数据复制到临时向量的对应桶中。
      for (int i = 0; i < (int)cur_p->size; ++i) {
        int j = bucket_idx[i]; // 获取当前 token 的桶索引。
        if (j >= ib) {         // 如果该桶在需要保留的范围内。
          // 将 token 复制到对应的桶，并更新桶指针。
          *bucket_ptrs[nbuckets - 1 - j]++ = cur_p->data[i];
        }
      }

      ptr = tmp_tokens.data(); // 重新指向临时向量的起始。
      int ndone = 0;           // 已排序的 token 数量。

      // 对所有高于 'ib' 索引的桶进行完全排序。
      for (int j = nbuckets - 1; j > ib; --j) {
        std::sort(ptr, ptr + histo[j], comp); // 对当前桶内的 token 进行排序。
        ptr += histo[j];                      // ptr 移动到下一个桶的起始位置。
        ndone += histo[j];                    // 累加已排序的 token 数量。
      }
      // 对包含第 k 个元素的桶（索引为 'ib'）进行部分排序，只排序到 k-ndone
      // 个元素。
      std::partial_sort(ptr, ptr + k - ndone, ptr + histo[ib], comp);
      // 将排序后的前 k 个 token 复制回 cur_p->data。
      std::memcpy(cur_p->data, tmp_tokens.data(), k * sizeof(llama_token_data));
    }
    cur_p->sorted = true; // 标记数组已排序。
  }

  cur_p->size = k; // 将 cur_p->size 截断为 k，表示只保留了前 k 个 token。
}
} // namespace

// GreedySampling (贪心采样) 类的实现。
// 贪心采样总是选择 logit 最大的 token。
void GreedySampling::apply(llama_token_data_array *cur_p) {
  cur_p->selected = 0;                       // 假设第一个 token 是最大的。
  for (size_t i = 1; i < cur_p->size; ++i) { // 从第二个 token 开始遍历。
    // 如果当前 token 的 logit 大于目前选定的 token 的 logit。
    if (cur_p->data[i].logit > cur_p->data[cur_p->selected].logit) {
      cur_p->selected = i; // 更新选定 token 的索引。
    }
  }
}

DistSampling::DistSampling(uint32_t seed) : seed_(seed) {
  seed_cur_ = get_rng_seed(seed); // 获取实际的随机数种子。
  // 使用种子初始化 Mersenne Twister 随机数引擎。
  rng_ = std::mt19937(seed_cur_);
}

void DistSampling::apply(llama_token_data_array *cur_p) {
  // 首先对 logits 进行 Softmax 归一化，得到概率。
  llama_sampler_softmax_impl(cur_p);
  // 然后根据这些概率执行离散分布采样，选择一个 token。
  cur_p->selected = llama_sample_dist(cur_p, rng_);
}

// 重置随机数引擎的状态，通常用于确保可复现性或在新的生成周期开始时重置。
void DistSampling::reset() {
  seed_cur_ = get_rng_seed(seed_);
  rng_.seed(seed_cur_);
}

// TopKSampling (Top-K 采样) 的 apply 方法。
// 直接调用辅助函数 llama_sampler_top_k_impl 来实现 Top-K 过滤。
void TopKSampling::apply(llama_token_data_array *cur_p) {
  // 对 token 数组应用 Top-K 过滤，只保留前 k 个。
  llama_sampler_top_k_impl(cur_p, k_);
}

// TopPSampling (Top-P/Nucleus 采样) 的 apply 方法。
// Top-P 采样根据累积概率阈值 p 来动态确定候选 token 集合的大小。
void TopPSampling::apply(llama_token_data_array *cur_p) {
  if (p_ >= 1.0f) {
    // 如果 p >= 1.0，表示不进行 Top-P 过滤（保留所有 token）。
    return;
  }
  // 在选择之前，需要做两件事情：
  // 1.从大到小排序
  // 2.Softmax转换为概率分布
  llama_sampler_softmax_impl(cur_p);

  float cum_sum = 0.0f; // 累积概率和。
  size_t last_idx =
      cur_p->size; // 记录需要保留的最后一个 token 的索引（不包含）

  // 遍历 token 数组，累加概率，直到达到或超过 p。
  for (size_t i = 0; i < cur_p->size; ++i) {
    cum_sum += cur_p->data[i].p; // 累加当前 token 的概率。

    // 检查累积和是否达到 p，并且保留的 token 数量是否至少达到 min_keep。
    // 如果满足条件，则确定 last_idx，并跳出循环。
    if (cum_sum >= p_ && i + 1 >= min_keep_) {
      last_idx = i + 1; // 记录下边界，所有在这个索引之前的 token 都被保留。
      break;
    }
  }

  // 根据 last_idx 调整 cur_p->size，只保留 Top-P 范围内或满足 min_keep 的
  // token。
  cur_p->size = last_idx;
}

// MinPSampling (Min-P 采样) 的 apply 方法。
// Min-P 采样保留所有概率不低于 “最高概率 * p” 的 token。
void MinPSampling::apply(llama_token_data_array *cur_p) {
  if (p_ <= 0.0f || !cur_p->size) {
    // 如果 p <= 0 或没有候选 token，则不执行操作。
    return;
  }

  // 标志位，表示 Min-P 逻辑是否已成功应用。
  bool min_p_applied = false;

  // 如果当前数据未排序，尝试使用一个不需要完整排序的实现。
  // 这是一种优化，如果只需要过滤而不完全依赖排序顺序。
  if (!cur_p->sorted) {
    // 用于存储过滤后的 token。
    std::vector<llama_token_data> filtered_tokens;

    // 初始化最大 logit 为最小浮点数。
    float max_logit = -FLT_MAX;
    for (size_t i = 0; i < cur_p->size; ++i) {
      // 找到所有 token 中的最大 logit。
      max_logit = std::max(max_logit, cur_p->data[i].logit);
    }

    // 计算最小 logit 阈值：max_logit + log(p_)。
    // 因为 p = exp(logit) / sum_exp，所以 log(p) = logit - log(sum_exp)。
    // 如果要 p_i >= p * p_max，则 log(p_i) >= log(p * p_max) = log(p) +
    // log(p_max)。 在这里 p_max 对应 max_logit 对应的概率，所以 min_logit 对应
    // log(p * p_max)。
    const float min_logit =
        max_logit + logf(p_); // 目标是保留 logit >= max_logit + log(p)。

    for (size_t i = 0; i < cur_p->size; ++i) {
      if (cur_p->data[i].logit >= min_logit) {
        // 如果当前 token 的 logit 大于等于阈值。
        // 将其添加到过滤后的列表中。
        filtered_tokens.push_back(cur_p->data[i]);
      }
    }

    // 如果过滤后的 token 列表不为空且数量达到 min_keep，则应用成功。
    if (!filtered_tokens.empty() && filtered_tokens.size() >= min_keep_) {
      // 将过滤后的数据复制回 cur_p->data。
      memcpy(cur_p->data, filtered_tokens.data(),
             filtered_tokens.size() * sizeof(llama_token_data));
      // 更新 cur_p->size。
      cur_p->size = filtered_tokens.size();
      min_p_applied = true;
    }
  }

  // 如果上述未排序的优化未应用成功（比如数据已排序或过滤结果不满足 min_keep），
  // 则回退到基于排序的实现。
  if (!min_p_applied) {
    // 确保数据按 logit 降序排序。
    if (!cur_p->sorted) {
      std::sort(cur_p->data, cur_p->data + cur_p->size,
                [](const llama_token_data &a, const llama_token_data &b) {
                  return a.logit > b.logit;
                });
      cur_p->sorted = true;
    }
    // 计算最小 logit 阈值。由于已排序，cur_p->data[0].logit 是最大的。
    // 目标是保留 logit >= max_logit + log(p)。
    const float min_logit = cur_p->data[0].logit + logf(p_);
    size_t i = 1; // 从第二个 token 开始检查（第一个 token 必然满足条件）。

    // 遍历已排序的 token，找到第一个 logit 小于阈值并且已保留数量满足 min_keep
    // 的位置。
    for (; i < cur_p->size; ++i) {
      if (cur_p->data[i].logit < min_logit && i >= min_keep_) {
        break; // 如果 logit 太小且已满足 min_keep，则停止。
      }
    }

    // 调整 cur_p->size，只保留满足条件的 token。
    cur_p->size = i;
  }
}

// TypicalSampling (典型采样) 的 apply 方法。
// 典型采样基于信息熵来过滤“非典型”的 token。
void TypicalSampling::apply(llama_token_data_array *cur_p) {
  // Reference implementation:
  // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
  if (p_ >= 1.0f) {
    // 如果 p >= 1.0，不执行过滤。
    return;
  }

  // 首先计算 Softmax 概率和整个分布的熵。
  // 计算概率 p，并确保已按 logit 降序排序。
  llama_sampler_softmax_impl(cur_p);

  float entropy = 0.0f; // 存储熵值。
  // 计算熵：-Σ p * log(p)。
  for (size_t i = 0; i < cur_p->size; ++i) {
    entropy += -cur_p->data[i].p * logf(cur_p->data[i].p);
  }

  // 计算每个 token 的“信息量”（-log(p)）与平均信息量（熵）的绝对差。
  std::vector<float> shifted_scores;
  for (size_t i = 0; i < cur_p->size; ++i) {
    // |(-log(p)) - entropy|。
    float shifted_score = fabsf(-logf(cur_p->data[i].p) - entropy);
    shifted_scores.push_back(shifted_score);
  }

  // 创建索引向量并根据 shifted_scores 对其排序。
  // 目的是找到 shifted_scores 最小的 token（即最“典型”的 token）。
  std::vector<size_t> indices(cur_p->size);
  // 用 0 到 cur_p->size-1 填充索引向量。
  std::iota(indices.begin(), indices.end(), 0);

  // 根据 shifted_scores 对索引进行排序。升序
  std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
    // shifted_scores 越小越靠前。
    return shifted_scores[a] < shifted_scores[b];
  });

  float cum_sum = 0.0f; // 累积概率和。
  // 记录需要保留的最后一个 token 的索引（不包含）。
  size_t last_idx = indices.size();

  // 遍历排序后的索引，累加对应 token 的概率。
  for (size_t i = 0; i < indices.size(); ++i) {
    size_t idx = indices[i];       // 获取当前“最典型”的 token 的原始索引。
    cum_sum += cur_p->data[idx].p; // 累加对应 token 的概率。

    // 如果累积和超过 typical 阈值 p，并且已保留的 token 数量满足 min_keep。
    if (cum_sum > p_ && (min_keep_ == 0 || i >= min_keep_ - 1)) {
      last_idx = i + 1; // 确定保留的边界。
      break;
    }
  }

  // 根据筛选出的索引构建新的 token 列表。
  std::vector<llama_token_data> cur_p_new;
  for (size_t i = 0; i < last_idx; ++i) {
    size_t idx = indices[i];
    cur_p_new.push_back(cur_p->data[idx]); // 将筛选出的 token 添加到新列表。
  }

  // 将新列表的数据复制回 cur_p->data。
  std::copy(cur_p_new.begin(), cur_p_new.end(), cur_p->data);
  cur_p->size = cur_p_new.size(); // 更新 cur_p->size。
  cur_p->sorted = false;          // 由于重新排列，标记为未排序。
}

// TemperatureSampling (温度采样) 的 apply 方法。
// 直接调用辅助函数 llama_sampler_temp_impl 来应用温度。
void TemperatureSampling::apply(llama_token_data_array *cur_p) {
  llama_sampler_temp_impl(cur_p, temp_);
}

// TemperatureExSampling (动态温度采样) 的 apply 方法。
// 动态温度采样根据分布的熵来调整温度。
void TemperatureExSampling::apply(llama_token_data_array *cur_p) {
  if (delta_ > 0) {
    // 如果 delta > 0，表示启用动态温度。
    const float min_temp = (std::max)(0.0f, temp_ - delta_); // 最小温度。
    const float max_temp = temp_ + delta_;                   // 最大温度。

    float exponent_val = exponent_; // 温度调整的指数。

    // 如果只有一个或没有候选 token，则无需进行温度调整。
    if (cur_p->size <= 1) {
      return;
    }

    // 计算最大可能的熵（均匀分布时的熵）。均匀分布时，熵最大
    // H_max = -log(1/N) = log(N)。
    float max_entropy = -logf(1.0f / cur_p->size);

    // 首先对 logits 进行 Softmax 归一化，得到概率。
    llama_sampler_softmax_impl(cur_p);

    // 计算当前 Softmax 概率分布的熵。
    float entropy = 0.0f;
    for (size_t i = 0; i < cur_p->size; ++i) {
      float prob = cur_p->data[i].p;
      if (prob > 0.0f) { // 避免 log(0) 的情况。
        entropy -= prob * logf(prob);
      }
    }

    // 归一化熵值（将当前熵值映射到 [0, 1] 范围）。
    // max_entropy 在 cur_p->size > 1 时不会为 0。
    float normalized_entropy = entropy / max_entropy;

    // 使用幂函数将归一化熵映射到动态温度范围。
    // dyn_temp = min_temp + (max_temp - min_temp) * (normalized_entropy ^
    // exponent_val)。
    float dyn_temp = min_temp + (max_temp - min_temp) *
                                    powf(normalized_entropy, exponent_val);

    // 使用动态计算出的温度对 logits 进行调整。
    llama_sampler_temp_impl(cur_p, dyn_temp);

    // 在应用动态温度后，logits 已经改变，需要重新计算 Softmax 概率。
    // 再次获取最大 logit，以避免溢出。
    const double max_l_double = cur_p->data[0].logit;

    double cum_sum_double = 0.0;
    for (size_t i = 0; i < cur_p->size; ++i) {
      double p = exp(cur_p->data[i].logit - max_l_double); // 计算未归一化概率。
      cur_p->data[i].p = p;                                // 存储未归一化概率。
      cum_sum_double += p;                                 // 累加。
    }

    for (size_t i = 0; i < cur_p->size; ++i) {
      cur_p->data[i].p /= cum_sum_double; // 重新归一化概率。
    }
  } else {
    // 如果 delta <= 0，则退化为普通温度采样。
    llama_sampler_temp_impl(cur_p, temp_);
  }
}

// XTCSampling (Exclude The Top Choice 采样) 的构造函数。
XTCSampling::XTCSampling(float p, float threshold, size_t min_keep,
                         uint32_t seed)
    : probability_(p), threshold_(threshold), min_keep_(min_keep), seed_(seed) {
  seed_cur_ = get_rng_seed(seed);
  rng_ = std::mt19937(seed_cur_);
}

// XTCSampling 的 apply 方法。
// 以一定概率移除概率最高的 token，鼓励多样性。
// "以一定概率":就是有时候过滤，有时候不过滤
void XTCSampling::apply(llama_token_data_array *cur_p) {
  if (probability_ <= 0.0f || threshold_ > 0.5f || cur_p->size < 2) {
    // 检查无效参数或不足的候选 token。
    return;
  }

  // 0到1之间的均匀分布。
  std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  float chance = distribution(rng_); // 生成一个随机数。
  if (chance > probability_) // 如果随机数大于设定的概率，则不执行排除操作。
    return;

  // 确保数据已排序并计算了 Softmax 概率。
  llama_sampler_softmax_impl(cur_p);

  int pos_last = 0; // 记录最后一个概率大于等于阈值的 token 的索引。

  // 遍历已排序的 token，找到第一个概率小于阈值的 token。
  for (size_t i = 0; i < cur_p->size; ++i) {
    if (cur_p->data[i].p >= threshold_) { // 如果当前 token 的概率大于等于阈值。
      pos_last = i;                       // 更新 pos_last。
    } else
      break; // 如果小于阈值，由于已排序，后续 token 也会小于阈值，所以停止。
  }

  // 如果剩余的 token 数量足够（大于 min_keep_）且移除的 token 不为 0。
  if (cur_p->size - pos_last >= min_keep_ && pos_last > 0) {
    cur_p->data += pos_last; // 移动指针，跳过概率最高的 pos_last 个 token。
    cur_p->size -= pos_last; // 更新数组大小。
  }
}

// XTCSampling 的 reset 方法。
// 重置随机数引擎的状态。
void XTCSampling::reset() {
  seed_cur_ = get_rng_seed(seed_);
  rng_.seed(seed_cur_);
}

// TopNSigmaSampling (Top-N-Sigma 采样) 的 apply 方法。
// Top-N-Sigma 采样根据 logits 的均值和标准差来过滤“异常”token。
void TopNSigmaSampling::apply(llama_token_data_array *cur_p) {
  if (n_ <= 0.0f || cur_p->size <= 1) {
    // 如果 n 小于等于 0 或候选 token 不足，不执行操作。
    return;
  }

  // 找到最大 logit 并计算均值。
  float max = cur_p->data[0].logit; // 假设第一个 logit 是最大的。
  float logits_sum = 0;             // logit 总和。
  size_t valid_count = 0;           // 有效（非负无穷）logit 的数量。
  for (size_t i = 0; i < cur_p->size; ++i) {
    // 只计算非负无穷的 logit。
    if (cur_p->data[i].logit != -INFINITY) {
      if (cur_p->data[i].logit > max) {
        max = cur_p->data[i].logit; // 更新最大 logit。
      }
      logits_sum += cur_p->data[i].logit; // 累加 logit。
      valid_count++;                      // 增加有效计数。
    }
  }

  // 计算均值。如果 valid_count 为 0，则均值为 0。
  float mean = valid_count > 0 ? logits_sum / valid_count : 0;

  // 计算标准差。
  float acc = 0; // 累加平方差。
  for (size_t i = 0; i < cur_p->size; ++i) {
    // 忽略负无穷
    if (cur_p->data[i].logit != -INFINITY) {
      // 累加 (logit - 均值)^2。
      acc += pow(cur_p->data[i].logit - mean, 2);
    }
  }
  // 计算标准差。如果 valid_count 为 0，则标准差为 0。
  float std = valid_count > 0 ? sqrt(acc / valid_count) : 0;

  // 应用掩码：将 logit 值低于 `max - (n_ * std)` 的 token 设置为负无穷大。
  for (size_t i = 0; i < cur_p->size; ++i) {
    if (cur_p->data[i].logit < max - (n_ * std)) {
      // 如果 logit 低于阈值。
      // 将其 logit 设置为负无穷，使其被排除。
      cur_p->data[i].logit = -INFINITY;
    }
  }
  // 重新计算 Softmax 概率，因为部分 logit 已经改变。
  llama_sampler_softmax_impl(cur_p);
}

// MirostatSampling (Mirostat v1 采样) 的构造函数。
MirostatSampling::MirostatSampling(int32_t n_vocab, uint32_t seed, float tau,
                                   float eta, int32_t m)
    : n_vocab_(n_vocab), seed_(seed), tau_(tau), eta_(eta), m_(m),
      mu_(2.0f * tau) { // mu_ 的初始值通常设置为 2 * tau。
  seed_cur_ = get_rng_seed(seed);
  rng_ = std::mt19937(seed_cur_);
}

// MirostatSampling 的 apply 方法。
// Mirostat v1 是一种自适应的 Top-K 采样，旨在维持恒定的“惊奇度”。
void MirostatSampling::apply(llama_token_data_array *cur_p) {
  // 首先计算 Softmax 概率，并确保已排序。
  llama_sampler_softmax_impl(cur_p);

  // 估计 s_hat，这是一种基于 Top-M token 概率梯度的“熵”估计。
  // s_hat 用于估算当前概率分布的平坦程度。
  float s_hat = 0.0;
  float sum_ti_bi = 0.0;
  float sum_ti_sq = 0.0;
  // 遍历前 m-1 个 token 对（i 和 i+1）。
  for (size_t i = 0; i < size_t(m_ - 1) && i < cur_p->size - 1; ++i) {
    // t_i 和 b_i 是用于线性回归估计 s_hat 的项。
    float t_i = logf(float(i + 2) / float(i + 1));
    // 相邻 token 概率的比值的对数。
    float b_i = logf(cur_p->data[i].p / cur_p->data[i + 1].p);
    sum_ti_bi += t_i * b_i; // 累加 t_i * b_i。
    sum_ti_sq += t_i * t_i; // 累加 t_i 的平方。
  }
  // 线性回归斜率的估计。
  s_hat = sum_ti_bi / sum_ti_sq;

  // 根据 s_hat 和目标惊奇度 tau 计算动态的 k 值。
  // 这个公式来自 Mirostat 论文，它将 s_hat 转换为一个合适的 k
  // 值，以维持目标惊奇度。
  float epsilon_hat = s_hat - 1;
  float k =
      powf((epsilon_hat * powf(2, mu_)) / (1 - powf(n_vocab_, -epsilon_hat)),
           1 / s_hat);

  // 应用计算出的 k 值进行 Top-K 过滤。
  llama_sampler_top_k_impl(cur_p, std::max(int(k), 1)); // 确保 k 至少为 1。
  llama_sampler_softmax_impl(
      cur_p); // 再次计算 Softmax 概率，因为 Top-K 过滤改变了候选集。

  // 从过滤后的分布中采样一个 token。
  const int idx = llama_sample_dist(cur_p, rng_);

  cur_p->selected = idx; // 记录选定的 token 索引。

  // 计算实际观察到的惊奇度。S = -log2(p)。
  float observed_surprise = -log2f(cur_p->data[idx].p);
  // 误差 = 实际惊奇度 - 目标惊奇度。
  float e = observed_surprise - tau_;

  // 更新 mu（Mirostat 的内部状态，代表“注意力”或“焦点”）。
  // mu_ = mu_ - eta_ * e， eta_ 是学习率。通过反馈调整
  // mu，使实际惊奇度趋近目标惊奇度。
  mu_ = mu_ - eta_ * e;
}

// MirostatSampling 的 reset 方法。
// 重置 Mirostat 内部状态。
void MirostatSampling::reset() {
  mu_ = 2.0f * tau_; // 重置 mu_ 为初始值。
  seed_cur_ = get_rng_seed(seed_);
  rng_.seed(seed_cur_);
}

// MirostatV2Sampling (Mirostat v2 采样) 的构造函数。
MirostatV2Sampling::MirostatV2Sampling(uint32_t seed, float tau, float eta)
    : seed_(seed), tau_(tau), eta_(eta_), mu_(2.0f * tau) {
  seed_cur_ = get_rng_seed(seed);
  rng_ = std::mt19937(seed_cur_);
}

// MirostatV2Sampling 的 apply 方法。
// Mirostat v2 是一种更直接的惊奇度控制方法。
void MirostatV2Sampling::apply(llama_token_data_array *cur_p) {
  // 首先计算 Softmax 概率，并确保已排序。
  llama_sampler_softmax_impl(cur_p);

  // 截断所有惊奇度大于 mu 的 token。
  // std::find_if 找到第一个满足条件的 token，然后截断到该位置。
  cur_p->size = std::distance(
      cur_p->data, std::find_if(cur_p->data, cur_p->data + cur_p->size,
                                [&](const llama_token_data &candidate) {
                                  // 惊奇度 S = -log2(p)
                                  return -log2f(candidate.p) > mu_;
                                }));
  // 如果截断后没有 token，至少保留一个。
  if (cur_p->size == 0) {
    cur_p->size = 1;
  }

  // 对剩余 token 的概率进行归一化（因为截断改变了总和）。
  llama_sampler_softmax_impl(cur_p);

  // 从过滤后的分布中采样一个 token。
  const int idx = llama_sample_dist(cur_p, rng_);

  cur_p->selected = idx; // 记录选定的 token 索引。

  // 计算实际观察到的惊奇度。
  float observed_surprise = -log2f(cur_p->data[idx].p);
  float e = observed_surprise - tau_; // 误差。

  // 通过反馈调整 mu_。
  mu_ = mu_ - eta_ * e;
}

// MirostatV2Sampling 的 reset 方法。
// 重置 Mirostat v2 内部状态。
void MirostatV2Sampling::reset() {
  mu_ = 2.0f * tau_;
  seed_cur_ = get_rng_seed(seed_);
  rng_.seed(seed_cur_);
}

// PenaltiesSampling (标准惩罚采样) 的构造函数。
PenaltiesSampling::PenaltiesSampling(int32_t penalty_last_n,
                                     float penalty_repeat, float penalty_freq,
                                     float penalty_present)
    : penalty_last_n_(
          std::max(penalty_last_n, 0)), // 确保 penalty_last_n 不为负。
      penalty_repeat_(penalty_repeat), penalty_freq_(penalty_freq),
      penalty_present_(penalty_present), prev_(penalty_last_n_) {
  // 如果 penalty_last_n_ 为 0，环形缓冲区容量设为 1，但实际不会使用。
  // 初始化一个环形缓冲区，用于存储最近的 N 个 token。
}

// PenaltiesSampling 的 apply 方法。
// 对重复出现的 token 施加惩罚，以减少重复。
void PenaltiesSampling::apply(llama_token_data_array *cur_p) {
  // 如果惩罚禁用，或者所有惩罚系数都为默认值（不生效），则直接返回。
  if ((penalty_last_n_ == 0) ||
      (penalty_repeat_ == 1.0f && penalty_freq_ == 0.0f &&
       penalty_present_ == 0.0f)) {
    return;
  }

  // 遍历所有候选 token，并根据其在 recent token 中的出现情况施加惩罚。
  for (size_t i = 0; i < cur_p->size; ++i) {
    // 在频率图中查找当前 token。
    const auto token_iter = token_count_.find(cur_p->data[i].id);
    if (token_iter == token_count_.end()) {
      // 如果 token 未在 recent token 中出现，则跳过。
      continue;
    }

    // 获取 token 的出现次数。
    const int count = token_iter->second;

    // 断言：确保计数在有效范围内。
    assert(count > 0 && count <= penalty_last_n_);

    // 应用重复惩罚 (penalty_repeat)。
    // 原始论文中是除法，但会导致负 logit 变得更正，错误地增加概率。
    // 这里的实现是：如果 logit <= 0，则乘以惩罚因子；如果 logit >
    // 0，则除以惩罚因子。 这样无论 logit
    // 正负，其绝对值都会减小，从而降低其概率。
    if (cur_p->data[i].logit <= 0) {
      // logit <= 0，乘以惩罚因子（通常 > 1
      // 会使其绝对值变大，但这里应该是使其更小，所以 penalty_repeat_ 应该在 0-1
      // 之间？需要确认）。 实际的 llama.cpp 逻辑是 repeat penalty 大于 1.0
      // 时降低概率，所以这里是当 logit <= 0 时，log(p) = logit / T 会因为 logit
      // 变小而变小，所以应该是乘以小于 1 的 penalty_repeat 或者 logit /=
      // penalty_repeat。 查阅 llama.cpp 源码，其 `penalty_repeat_`
      // 是一个大于等于 1.0 的值。 `logit /= penalty_repeat_` 会使得正的 logit
      // 变小，负的 logit 绝对值变小（更接近0）。 `logit *= penalty_repeat_`
      // 会使得负的 logit 绝对值变大（更负），正的 logit 变大。
      // 当前代码的逻辑是：如果 logit <= 0，则 `logit *= penalty_repeat_`。假设
      // `penalty_repeat_` > 1，负 logit 会变得更负，降低概率。 如果 logit >
      // 0，则 `logit /= penalty_repeat_`。正 logit 会变小，降低概率。
      // 这个实现是正确的，因为它统一地降低了重复 token 的概率。
      cur_p->data[i].logit *= penalty_repeat_;
    } else {
      // logit > 0，除以惩罚因子。
      cur_p->data[i].logit /= penalty_repeat_;
    }
    // 应用频率惩罚 (penalty_freq) 和存在惩罚 (penalty_present)。
    // 频率惩罚与出现次数成正比。存在惩罚只要出现过就施加。
    // 频率惩罚：count * penalty_freq。
    // 存在惩罚：如果 count > 0，则加上 penalty_present。
    cur_p->data[i].logit -=
        float(count) * penalty_freq_ + float(count > 0) * penalty_present_;
  }
  // 由于 logits 改变，标记为未排序。
  cur_p->sorted = false;
}

// PenaltiesSampling 的 reset 方法。
// 清空历史 token 记录和频率计数
void PenaltiesSampling::reset() {
  prev_.clear();
  token_count_.clear();
}

// PenaltiesSampling 的 accept 方法。
// 在采样并选择一个 token 后调用，将该 token 添加到历史记录中。
void PenaltiesSampling::accept(llama_token token) {
  if (penalty_last_n_ == 0) {
    // 如果惩罚禁用，则不记录历史 token。
    return;
  }

  token_count_[token]++; // 增加当前 token 的计数。

  // 如果环形缓冲区已满（即已记录了 penalty_last_n_ 个 token）。
  if (prev_.size() >= (size_t)penalty_last_n_) {
    const auto old = prev_.front(); // 获取最旧的 token。

    token_count_[old]--; // 减少最旧 token 的计数。
    if (token_count_[old] == 0) {
      token_count_.erase(old); // 如果计数为 0，从图中移除该 token。
    }
  }

  prev_.push_back(token);
}

// DrySampling (Don't Repeat Yourself 采样) 的构造函数。
DrySampling::DrySampling(
    int32_t context_size, float dry_multiplier, float dry_base,
    int32_t dry_allowed_length, int32_t dry_penalty_last_n,
    const std::unordered_multimap<llama_token, std::vector<llama_token>>
        &processed_breakers)
    : total_context_size_(context_size), dry_multiplier_(dry_multiplier),
      dry_base_(dry_base), dry_allowed_length_(dry_allowed_length),
      dry_penalty_last_n_(dry_penalty_last_n),
      last_tokens_(ring_buffer<llama_token>(0)) {
  // 计算实际的惩罚范围，-1 表示整个上下文。
  int32_t effective_dry_penalty_last_n = (dry_penalty_last_n == -1)
                                             ? context_size
                                             : std::max(dry_penalty_last_n, 0);
  // 判断 DrySampling 是否启用。
  const bool dry_enabled =
      (dry_multiplier != 0.0f && dry_base >= 1.0f && dry_penalty_last_n != 0);

  // 根据是否启用 DrySampling 来初始化 `dry_repeat_count_` 和 `last_tokens_`。
  dry_repeat_count_ = dry_enabled
                          ? std::vector<int>(effective_dry_penalty_last_n, 0)
                          : std::vector<int>{},
  last_tokens_ = dry_enabled
                     ? ring_buffer<llama_token>(effective_dry_penalty_last_n)
                     : ring_buffer<llama_token>(0);
  if (dry_enabled) {
    // 只有启用时才复制断点序列。
    dry_processed_breakers_ = processed_breakers;
  }
}

// Ported from Koboldcpp, original PR:
// https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
// DrySampling 的 apply 方法。
// 这是一种更复杂的重复惩罚，旨在识别和阻止长序列的重复。
void DrySampling::apply(llama_token_data_array *cur_p) {
  if (dry_multiplier_ == 0.0f || dry_base_ < 1.0f || dry_penalty_last_n_ == 0) {
    // 如果 DrySampling 禁用，则直接返回。
    return;
  }

  // 计算实际的惩罚范围，确保不超过上下文大小和环形缓冲区大小。
  int32_t effective_dry_penalty_last_n = (dry_penalty_last_n_ == -1)
                                             ? total_context_size_
                                             : std::max(dry_penalty_last_n_, 0);
  int last_n_repeat =
      std::min(std::min((int)last_tokens_.size(), effective_dry_penalty_last_n),
               total_context_size_);

  // 如果最近的 token 数量不足以触发惩罚，则返回。
  if (last_n_repeat <= dry_allowed_length_) {
    return;
  }
  // 初始化重复计数器。
  dry_repeat_count_.assign(last_n_repeat, 0);
  dry_max_token_repeat_.clear(); // 清空最大 token 重复记录。

  // 步骤 1：查找重启序列以限制最大重复长度。
  // 从上下文（历史文本）中向后查找，寻找任何作为重启序列开头（“头”令牌）的令牌。
  //
  // 集合 `restart_sequences`
  // 是一个映射，它将“头”令牌映射到所有构成重启序列的“尾部”序列。
  // 这使我们能够快速检查每个令牌是否是完整序列的开头。大多数重启序列实际上都是
  // 单个令牌，对于这些情况，“尾部”是一个空向量。
  //
  // 如果一个令牌是“头”，则测试所有以该令牌开始的重启序列
  // （通常每个令牌只有一个序列，但如果使用像“aaaq1”和“aaa1”这样的序列作为重启字符串，
  // 当它们被分词时，两者都可能以“aaa”开头）。最长匹配序列（如果有的话）用于
  // 限制最大重复长度。
  //
  // 请注意，如果一个短序列包含在一个长序列中，这可能无法找到 `rep_limit`
  // 的最小值。
  // 例如，如果“羊水”（amniotic）和“尼”（ni）都被用作重启序列，“尼”将被首先找到，
  // 由于它较短，它将无法抑制“otic”。这是一个小问题，因为完全包含的重启序列可能很少见。
  //
  // 理论上，对于任意重启序列，这在最坏情况下是 O(N^2) 的复杂度，
  // 这就是为什么我们在生成 `restart_sequences` 时已经限制了最大尾部序列长度。
  // 通过限制，这个扫描在上下文长度上是 O(N) 的。
  int rep_limit = last_n_repeat; // 默认最大重复长度是所有 recent token。
  for (int i = 0; i < last_n_repeat; ++i) {
    llama_token token =
        last_tokens_.rat(i); // 获取环形缓冲区中倒数第 i 个 token。
    auto its = dry_processed_breakers_.equal_range(
        token); // 查找以该 token 为开头的重启序列。
    if (its.first == dry_processed_breakers_.end()) {
      // 如果没有找到以该 token 开头的重启序列。
      continue;
    }
    int longest_match = -1; // 记录最长匹配的重启序列的长度。
    for (auto it = its.first; it != its.second; ++it) {
      // seq_len 是重启序列中“尾部”token 的数量。
      int seq_len = (int)it->second.size();
      if (seq_len > longest_match &&
          seq_len <= (int)i) { // 如果当前序列更长且在当前遍历范围内。
        bool match = true;
        // 检查当前上下文是否与重启序列的“尾部”匹配。
        for (int offset = 0; offset < seq_len; ++offset) {
          // last_tokens_.rat(i - offset - 1) 获取的是“头”token 之前的 token。
          if (it->second[offset] != last_tokens_.rat(i - offset - 1)) {
            match = false;
            break;
          }
        }
        if (match) {
          longest_match = seq_len; // 更新最长匹配长度。
        }
      }
    }
    if (longest_match >= 0) {
      // 如果找到了重启序列，则更新 rep_limit。
      // rep_limit = 当前 token 离末尾的距离 -
      // 重启序列的长度，即在重启序列之前的部分不考虑重复。
      rep_limit = i - longest_match;
      break;
    }
  }
  // 如果新的 rep_limit 小于允许的重复长度，则不执行惩罚。
  if (rep_limit < dry_allowed_length_) {
    return;
  }

  // 步骤 2：对上下文的最后 N 个令牌进行反向迭代，
  // 使用“Z-算法”（反向）高效计算上下文中其他位置出现的后缀的位置和长度。
  // 我们将后缀长度限制在 `rep_limit` 以尊重重启序列。
  //
  // 该算法目前在维基百科上没有文档，但这里有一个清晰的描述：
  // https://ivanyu.me/blog/2014/10/15/z-algorithm/
  //
  // 以下代码改编自同一作者的公共领域实现：
  // https://github.com/ivanyu/string-algorithms/blob/master/z_algorithm.py
  //
  // 示例：
  // 最后 N 个令牌：a b c c b c y a b c
  // 重复计数：     0 0 3 1 0 2 0 0 0 0
  //                    ^
  //   这个 `3` 意味着上下文的最后三个令牌（a b c）也出现在这里。
  //
  // 尽管看起来有嵌套的 for/while 循环，但这一步在最坏情况下是 O(N) 的，
  // 因为 Z-算法是线性的。这可以通过观察 `lt` 和 `rt` 边界在每次检测到重复后缀后
  // （即在每次 `n > 0` 的 while 循环之后）都会被设置来证实。
  // 这些边界变量确保内部的 while 循环在外部 for 循环迭代上下文时，
  // 只检查上下文中的每个令牌一次。

  {
    const int last = last_n_repeat - 1; // 最后一个 token 的索引。
    int rt = 0, lt = 0;                 // Z-box 的右边界和左边界。

    for (int k = 1; k < last_n_repeat; ++k) { // 遍历所有可能的起始位置。
      if (k > rt) {
        // 如果 k 超出当前 Z-box，执行朴素匹配。
        int n = 0; // 匹配长度。
        while (n + k < last_n_repeat &&
               last_tokens_.rat(n) == last_tokens_.rat(n + k)) {
          ++n; // 匹配成功，增加长度。
        }
        // dry_repeat_count_[last - k] 存储的是以当前位置 k
        // 开始，向左延伸的重复长度。
        dry_repeat_count_[last - k] = std::min(n, rep_limit);
        if (n > 0) {
          lt = k;         // 更新 Z-box 左边界。
          rt = k + n - 1; // 更新 Z-box 右边界。
        }
      } else {
        // 如果 k 在当前 Z-box 内部，利用 Z-box 的性质进行优化。
        int p = k - lt;                  // 对应 Z-box 内的对称位置。
        int right_part_len = rt - k + 1; // Z-box 右半部分的长度。

        if (dry_repeat_count_[last - p] < right_part_len) {
          // 如果对称位置的 Z 值小于右半部分长度，则直接使用对称位置的 Z 值。
          int n = std::min(dry_repeat_count_[last - p], rep_limit);
          dry_repeat_count_[last - k] = n;
        } else {
          // 如果对称位置的 Z 值大于或等于右半部分长度，需要向外扩展。
          int i = rt + 1; // 从 Z-box 右边界之外开始检查。
          while (i < last_n_repeat &&
                 last_tokens_.rat(i) == last_tokens_.rat(i - k)) {
            i += 1; // 扩展匹配长度
          }

          int n = std::min(i - k, rep_limit); // 计算实际匹配长度。
          dry_repeat_count_[last - k] = n;
          lt = k;     // 更新 Z-box 左边界。
          rt = i - 1; // 更新 Z-box 右边界。
        }
      }
    }
  }

  // 步骤 3：遍历 `dry_repeat_count` 和 `last_tokens`，
  // 检查通过发出每个新令牌（如果该令牌会延长序列）可能生成的最大重复长度。
  //
  // 沿用上面的例子：
  // 最后 N 个令牌：a b c c b c y a b c
  // 重复计数：      0 0 3 1 0 2 0 0 0 0
  //
  // 对于每个非零值，向前看一个令牌。如果这个令牌被发出，它将延长重复。
  // c: 3 -> 4（从 `a b c` 到 `a b c c`）
  // b: 1 -> 2（从 `c` 到 `c b`）
  // y: 2 -> 3（从 `b c` 到 `b c y`）

  for (int i = 0; i < last_n_repeat - 1; ++i) {
    int repeat_len = dry_repeat_count_[i]; // 获取当前位置的重复长度。
    if (repeat_len >= dry_allowed_length_) {
      // 如果重复长度超过允许值。
      // token = last_tokens_.rat(last_n_repeat - 2 - i)
      // 这个索引计算方式是为了获取在当前重复序列之后，下一个可能被生成的 token
      // 的 id。 dry_repeat_count_[i] 是 `rat(i)`
      // 后面匹配的长度，现在要惩罚下一个 token，这个 token 是 `rat(i-1)`
      // 对应的。 `last_tokens_.rat(idx)` 获取的是倒数第 `idx` 个 token。
      // `last_n_repeat - 1` 是 `rat(0)` 的实际位置。
      // `last_n_repeat - 1 - (i+1)` 或者 `last_n_repeat - 2 - i`
      llama_token token = last_tokens_.rat(last_n_repeat - 2 - i);
      // 记录每个 token 如果被生成，它能延续的最大重复长度。
      const auto &it = dry_max_token_repeat_.find(token);
      if (it == dry_max_token_repeat_.end() || it->second < repeat_len) {
        dry_max_token_repeat_[token] = repeat_len;
      }
    }
  }

  // 步骤 4：根据相关token的最大重复长度，应用 logit 惩罚。
  // 通过限制在 `max_exponent` 来防止 `pow(penalty_base, exponent)` 中
  // 的浮点数溢出。`max_exponent` 是根据 `penalty_base` 和
  // `std::numeric_limits<float>::max()` 的近似对数计算得出的。
  const float FLOAT_MAX_LOG = 88.7228391f; // log(FLT_MAX) 的近似值。
  int max_exponent = 0;
  if (dry_base_ > 1.000001f) {
    // 确保 dry_base_ 大于 1，否则 log(dry_base_) 为 0 或负数。
    max_exponent = FLOAT_MAX_LOG / std::log(dry_base_);
  }

  for (size_t i = 0; i < cur_p->size; ++i) {
    const auto &af_kvp = dry_max_token_repeat_.find(cur_p->data[i].id);
    if (af_kvp != dry_max_token_repeat_.end()) {
      // 检查所有以当前 token 开头的断点序列。
      auto range = dry_processed_breakers_.equal_range(cur_p->data[i].id);
      bool is_single_token_breaker = false;

      // 判断当前 token 是否是单个 token 的断点序列（例如，句号）。
      // 对于这些断点，即使它可能延续重复，通常也不应惩罚，因为它是正常的句子结构结束。
      for (auto it = range.first; it != range.second; ++it) {
        if (it->second.empty()) { // 如果尾部序列为空，说明是单 token 断点。
          is_single_token_breaker = true;
          break;
        }
      }

      // 只有当它不是单 token 断点序列时才应用惩罚。
      if (!is_single_token_breaker) {
        // 计算指数：重复长度 - 允许长度。
        int repeat_exp = af_kvp->second - dry_allowed_length_;
        if (max_exponent > 0 && repeat_exp > max_exponent) {
          repeat_exp = max_exponent; // 限制指数以避免溢出。
        }
        // 计算惩罚值：dry_multiplier * (dry_base ^ repeat_exp)。
        float penalty = dry_multiplier_ * std::pow(dry_base_, repeat_exp);
        cur_p->data[i].logit -= penalty; // 从 logit 中减去惩罚值。
      }
    }
  }
  // Logits 已改变，标记为未排序。
  cur_p->sorted = false;
}

// DrySampling 的 reset 方法。
// 清空所有内部状态。
void DrySampling::reset() {
  last_tokens_.clear();
  dry_repeat_count_.clear();
  dry_max_token_repeat_.clear();
}

// DrySampling 的 accept 方法。
// 在采样并选择一个 token 后调用，将其添加到历史记录中。
void DrySampling::accept(llama_token token) {
  // 如果 DrySampling 禁用，则不记录历史 token。
  if (dry_multiplier_ == 0.0f || dry_base_ < 1.0f || dry_penalty_last_n_ == 0) {
    return;
  }

  last_tokens_.push_back(token); // 将新采样的 token 添加到环形缓冲区。
}

void SamplingChain::add(std::unique_ptr<Sampling> sampling) {
  samples_.emplace_back(std::move(sampling));
}

void SamplingChain::llama_sampler_accept(llama_token token) {
  for (auto &i : samples_) {
    i->accept(token);
  }
}
void SamplingChain::llama_sampler_apply(llama_token_data_array *cur_p) {
  for (auto &i : samples_) {
    i->apply(cur_p);
  }
}
void SamplingChain::llama_sampler_reset() {
  for (auto &i : samples_) {
    i->reset();
  }
}

void SamplingChain::reset() { llama_sampler_reset(); }

llama_token SamplingChain::sample(llama_token_data_array *cur_p) {
  llama_sampler_apply(cur_p);
  assert(cur_p->selected >= 0 && cur_p->selected < (int32_t)cur_p->size);

  auto token = cur_p->data[cur_p->selected].id;

  llama_sampler_accept(token);

  return token;
}