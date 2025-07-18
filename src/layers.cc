#include "layers.h"
#include "cache.h"
#include "json.hpp"
#include "sampling.h"
#include <algorithm>
#include <fstream>
#include <iosfwd>
#include <iostream>
#include <math.h>
#include <random>
#include <string>

ModelConfig::ModelConfig()
    : vocab_size(50257), context_length(1024), emb_dim(768), n_heads(12),
      n_layers(12), drop_rate(0.1), qkv_bias(true) {}
ModelConfig::~ModelConfig() = default;
ModelConfig::ModelConfig(const ModelConfig &) = default;
ModelConfig &ModelConfig::operator=(const ModelConfig &) = default;

bool ModelConfig::InitFromFile(const std::string &config_file) {
  std::ifstream ifs(config_file);
  if (!ifs.is_open()) {
    std::cerr << "Failed to open config file: " << config_file << std::endl;
    return false;
  }

  nlohmann::json config_json;
  ifs >> config_json;

  // Parse JSON and initialize model config
  try {
    vocab_size = config_json.at("vocab_size").get<int64_t>();
    context_length = config_json.at("n_ctx").get<int64_t>();
    emb_dim = config_json.at("n_embd").get<int64_t>();
    n_heads = config_json.at("n_head").get<int64_t>();
    n_layers = config_json.at("n_layer").get<int64_t>();
    drop_rate = config_json.at("attn_pdrop").get<float>();
  } catch (const std::exception &e) {
    std::cerr << "Error parsing config file: " << e.what() << std::endl;
    return false;
  }

  return true;
}

Context::Context() = default;
Context::~Context() = default;

uint8_t *Context::allocate(size_t size) {
  if (!storage)
    return nullptr;
  return storage->allocate(size);
}

void Context::Register(Layer *layer) { layers.emplace_back(layer); }

Embedding::Embedding(Context *ctx, const std::string &instance_name,
                     int64_t num_embeddings, int64_t embedding_dim,
                     int64_t padding_idx)
    : Layer(ctx, instance_name), num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim), padding_idx_(padding_idx) {
  ctx_->Register(this);
  // WTE[50276,768],WPE[1024,768]
  W_ = dense::Tensor(dense::kFloat32, {num_embeddings_, embedding_dim_});
  grad_W_ = dense::Tensor::zeros(dense::kFloat32, W_.shape());
}

Embedding::~Embedding() = default;

void Embedding::LazyInit() {
  if (W_.empty()) {
    W_ = dense::Tensor::randn(W_.dtype(), W_.shape());
  }
}

dense::Tensor Embedding::forward(const dense::Tensor &input) {
  if (is_training()) {
    input_cache_ = input;
  }
  LazyInit();
  // input的形状限定为 [B,T]
  // 输出形状为 [B,T,C]
  // wte is (V,C) ->[50257,768]
  // wpe is (maxT,C)->[1024,768]

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = W_.size(-1);
  auto output = dense::Tensor(dense::kFloat32, {B, T, C});
  output.allocate();

  auto inp = reinterpret_cast<const int32_t *>(input.data());
  auto wp = reinterpret_cast<float *>(W_.data());
  auto outp = reinterpret_cast<float *>(output.data());

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      // out[b,t,:]
      float *out_bt = outp + b * T * C + t * C;
      int32_t ix = inp[b * T + t]; // inp[b, t]
      float *w_ix = wp + ix * C;
      for (size_t i = 0; i < C; ++i) {
        // 这里可用 std::memcpy替代，我们只是展示详细的计算过程
        out_bt[i] = w_ix[i];
      }
    }
  }
  return output;
}

dense::Tensor Embedding::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

Linear::Linear(Context *ctx, const std::string &instance_name,
               int64_t in_features, int64_t out_features, bool has_bias)
    : Layer(ctx, instance_name), in_features_(in_features),
      out_features_(out_features), has_bias_(has_bias) {
  ctx_->Register(this);
  // torch::nn::Linear 的权重形状是 [out_features, in_features]
  // 在 GPT-2 中，通常是 [2304,768]
  W_ = dense::Tensor(dense::kFloat32, {out_features_, in_features_});

  if (has_bias) {
    b_ = dense::Tensor(dense::kFloat32, {out_features_});
  }
}

Linear::~Linear() = default;

void Linear::LazyInit() {
  if (W_.empty()) {
    W_.allocate();
  }
  if (has_bias_ && b_.empty()) {
    b_.allocate();
  }
}

dense::Tensor Linear::forward(const dense::Tensor &input) {
  // 这里输入 input的形状是 [batch_size, seq_len,in_features]
  // 权重 weight_ 的形状是 [out_features, in_features]
  // 输出 output 的形状是 [batch_size, seq_len,out_features]

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = input.size(2);
  auto OC = W_.size(0); // 2304

  if (is_training()) {
    input_cache_ = input.clone();
  }
  LazyInit();

  auto output = dense::Tensor(input.dtype(), {B, T, OC});
  output.allocate();

  auto inp = reinterpret_cast<const float *>(input.data());
  auto wp = reinterpret_cast<float *>(W_.data());
  auto bp = reinterpret_cast<float *>(b_.data());
  auto outp = reinterpret_cast<float *>(output.data());

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      size_t bt = b * T + t;
      for (size_t o = 0; o < OC; ++o) {
        float val = has_bias_ ? bp[o] : 0.0f;
        for (size_t i = 0; i < C; ++i) {
          val += inp[bt * C + i] * wp[o * C + i]; // 这里包含转置
        }
        outp[bt * OC + o] = val;
      }
    }
  }
  return output;
}

dense::Tensor Linear::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

LayerNorm::LayerNorm(Context *ctx, const std::string &instance_name,
                     int64_t ndim, bool has_bias)
    : Layer(ctx, instance_name), ndim_(ndim), has_bias_(has_bias) {
  ctx_->Register(this);
  // ndim_ 是归一化的维度大小,在 GPT-2 中通常是 768
  // 参数名称: gpt2.transformer.h.11.ln_1.weight, 形状: [768], 类型: float
  // 参数名称: gpt2.transformer.h.11.ln_1.bias, 形状: [768], 类型: float
  W_ = dense::Tensor(dense::kFloat32, {ndim_});
  if (has_bias) {
    b_ = dense::Tensor(dense::kFloat32, {ndim_});
  }
}

LayerNorm::~LayerNorm() = default;

void LayerNorm::LazyInit() {
  if (W_.empty()) {
    W_.allocate();
  }
  if (has_bias_ && b_.empty()) {
    b_.allocate();
  }
}

dense::Tensor LayerNorm::forward(const dense::Tensor &input) {
  // 输入input的形状是 [batch_size, seq_len, ndim_]
  // ndim_ 是归一化的维度大小,在 GPT-2 中通常是 768
  // weight_ 的形状是 [ndim_], bias_ 的形状也是 [ndim_]
  // 输出 output 的形状与输入相同，即 [batch_size, seq_len, ndim_]
  if (is_training()) {
    input_cache_ = input.clone();
  }
  LazyInit();

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = input.size(2);

  auto output = dense::Tensor(input.dtype(), input.shape());
  output.allocate();

  mean_ = dense::Tensor(input.dtype(), {B, T});
  rstd_ = dense::Tensor(input.dtype(), {B, T});
  mean_.allocate();
  rstd_.allocate();

  auto inp = reinterpret_cast<const float *>(input.data());
  auto mean = reinterpret_cast<float *>(mean_.data());
  auto rstd = reinterpret_cast<float *>(rstd_.data());
  auto out = reinterpret_cast<float *>(output.data());
  auto weight = reinterpret_cast<float *>(W_.data());
  auto bias = reinterpret_cast<float *>(b_.data());

  float eps = 1e-5f;
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {

      auto x = inp + b * T * C + t * C;

      float m = 0.0f;
      for (size_t i = 0; i < C; ++i) {
        m += x[i];
      }
      m = m / C; // 均值

      float v = 0.0f;
      for (size_t i = 0; i < C; ++i) {
        float xshift = x[i] - m;
        v += xshift * xshift;
      }
      v = v / C; // 方差

      float s = 1.0f / std::sqrtf(v + eps);
      float *out_bt = out + b * T * C + t * C;
      for (size_t i = 0; i < C; ++i) {
        float n = (s * (x[i] - m));
        float o = n * weight[i] + bias[i];
        out_bt[i] = o;
      }
      mean[b * T + t] = m; // backward 要用到，避免重复计算
      rstd[b * T + t] = s; // backward 要用到，避免重复计算
    }
  }

  return output;
}

dense::Tensor LayerNorm::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

GELU::GELU(Context *ctx, const std::string &instance_name)
    : Layer(ctx, instance_name) {}
GELU::~GELU() = default;

#define GELU_SCALING_FACTOR sqrtf(2.0f / std::_Pi_val)

dense::Tensor GELU::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [batch_size, seq_len, emb_dim]
  // 输出 output 的形状与输入相同，即 [batch_size, seq_len, emb_dim]
  // GELU 激活函数的公式是: x * P(X <= x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x
  // + 0.044715 * x^3)))
  if (is_training()) {
    input_cache_ = input.clone();
  }
  auto B = input.size(0);
  auto T = input.size(1);
  auto C = input.size(2);

  auto N = B * T * C;

  auto output = dense::Tensor(input.dtype(), input.shape());
  output.allocate();

  auto inp = reinterpret_cast<const float *>(input.data());
  auto out = reinterpret_cast<float *>(output.data());
  for (size_t i = 0; i < N; ++i) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + std::tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
  return output;
}

dense::Tensor GELU::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

Dropout::Dropout(Context *ctx, const std::string &instance_name,
                 float dropout_ratio)
    : Layer(ctx, instance_name), dropout_ratio_(dropout_ratio) {
  if (dropout_ratio < 0.0 || dropout_ratio >= 1.0) {
    throw std::invalid_argument("dropout probability must be in [0, 1)");
  }
  scale_ = (dropout_ratio == 0.0) ? 1.0 : 1.0 / (1.0 - dropout_ratio);
}

Dropout::~Dropout() = default;

dense::Tensor Dropout::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [batch_size, seq_len, emb_dim]
  // 输出 output 的形状与输入相同，即 [batch_size, seq_len, emb_dim]
  // 在训练模式下，随机丢弃一部分神经元
  if (is_training()) {
    auto B = input.size(0);
    auto T = input.size(1);
    auto C = input.size(2);
    auto inp = reinterpret_cast<const float *>(input.data());
    mask_ = dense::Tensor::rand(input.dtype(), input.shape());
    auto mask = reinterpret_cast<float *>(mask_.data());
    auto output = dense::Tensor(input.dtype(), input.shape());
    output.allocate();
    auto out = reinterpret_cast<float *>(output.data());
    for (size_t b = 0; b < B; ++b) {
      for (size_t t = 0; t < T; ++t) {
        for (size_t c = 0; c < C; ++c) {
          size_t pos = b * T * C + t * C + c;
          mask[pos] = mask[pos] > dropout_ratio_ ? 1.0f : 0.0f;
          out[pos] = inp[pos] * mask[pos] * scale_;
        }
      }
    }
    return output;
  }
  return input;
}

dense::Tensor Dropout::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

MLP::MLP(Context *ctx, const std::string &instance_name,
         const ModelConfig &config)
    : Layer(ctx, instance_name), config_(config) {
  // config_.emb_dim 是嵌入维度,通常是 768
  c_fc_ = std::make_unique<Linear>(ctx, "c_fc", config_.emb_dim,
                                   4 * config_.emb_dim);
  gelu_ = std::make_unique<GELU>(ctx, "gelu");
  c_proj_ = std::make_unique<Linear>(ctx, "c_proj", 4 * config_.emb_dim,
                                     config_.emb_dim);
  dropout_ = std::make_unique<Dropout>(ctx, "dropout", config_.drop_rate);
}

MLP::~MLP() = default;

dense::Tensor MLP::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [batch_size, seq_len, emb_dim]
  // 输出 output 的形状也是 [batch_size, seq_len, emb_dim]
  auto x = c_fc_->forward(input);
  x = gelu_->forward(x);
  x = c_proj_->forward(x);
  x = dropout_->forward(x);
  return x;
}

dense::Tensor MLP::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

CausalSelfAttention::CausalSelfAttention(Context *ctx,
                                         const std::string &instance_name,
                                         const ModelConfig &config, size_t idx)
    : Layer(ctx, instance_name), config_(config), head_dim_(0), index_(idx) {
  assert(config_.emb_dim % config_.n_heads == 0);
  head_dim_ = config_.emb_dim / config_.n_heads;
  c_attn_ = std::make_unique<Linear>(ctx, "c_attn", config_.emb_dim,
                                     3 * config_.emb_dim, config_.qkv_bias);
  c_proj_ =
      std::make_unique<Linear>(ctx, "c_proj", config_.emb_dim, config_.emb_dim);
  attn_dropout_ =
      std::make_unique<Dropout>(ctx, "attn_dropout", config_.drop_rate);
  resid_dropout_ =
      std::make_unique<Dropout>(ctx, "resid_dropout", config_.drop_rate);
  mask_ = dense::Tensor::ones(dense::kInt8,
                              {config_.context_length, config_.context_length});
  {
    // 生成一个下三角掩码矩阵
    auto ptr = reinterpret_cast<int8_t *>(mask_.data());
    auto R = mask_.size(0);
    auto C = mask_.size(1);
    for (size_t i = 0; i < R; ++i) {
      for (size_t j = 0; j < C; ++j) {
        if (j > i)
          ptr[i * C + j] = 0;
      }
    }
  }
}

CausalSelfAttention::~CausalSelfAttention() = default;

dense::Tensor CausalSelfAttention::forward(const dense::Tensor &input) {
  if (!ctx_->training && ctx_->cache_) {
    return forward_cache(input);
  }
  // 在训练模式或者没有启用kv cache的情况下，我们处理前向传播

  auto B = input.size(0); // 批大小
  auto T = input.size(1); // 序列长度
  auto C = input.size(2); // 嵌入维度

  // input首先分别于Q,K,V相乘，生成 Q,K,V三个矩阵
  // 因为Q,K,V三个矩阵在合在一起的，我们做一次乘法就全部完成了
  // qkv [B,T,C*3]
  auto qkv = c_attn_->forward(input);
  float *qkv_ptr = reinterpret_cast<float *>(qkv.data());

  float scale = 1.0 / sqrtf(head_dim_);

  // 每个头att的形状都是 [T,T]
  auto att = dense::Tensor::zeros(dense::kFloat32, {T, T});
  auto att_ptr = reinterpret_cast<float *>(att.data());

  // 输出张量
  auto y = dense::Tensor::zeros(dense::kFloat32, {B, T, C});
  auto y_ptr = reinterpret_cast<float *>(y.data());

  auto mask_ptr = reinterpret_cast<int8_t *>(mask_.data());

  size_t C3 = C * 3;

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {

      // 计算每个头的注意力,我们的计算顺序与 llm.c不同，这里按照每个头独立计算
      // 1. q*k.transpose()
      for (size_t i = 0; i < T; ++i) {
        // 每个批次跨度 b * T * C3
        // 每个token跨度 C3
        // q_ptr指向当前批次，当前token，当前头的首指针
        float *q_ptr = qkv_ptr + b * T * C3 + i * C3 + h * head_dim_;

        for (size_t j = 0; j < T; ++j) {
          float *k_ptr =
              qkv_ptr + b * T * C3 + j * C3 + h * head_dim_ + C; // K的偏移是 C
          float val = 0.0f;
          // Q的行乘以K的列，这里已经包含转置
          for (size_t k = 0; k < head_dim_; ++k) {
            val += q_ptr[k] * k_ptr[k];
          }
          att_ptr[i * T + j] = val * scale;
        }
      }

      // mask fill, att[T,T]
      for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < T; ++j) {
          if (mask_ptr[i * config_.context_length + j] == 0) {
            att_ptr[i * T + j] = -INFINITY;
          }
        }
      }

      // softmax ,att[T,T]
      for (size_t i = 0; i < T; ++i) {
        // 先找行最大值，为了计算的数值稳定性，这是torch中的常规做法
        float maxval = -INFINITY; // 行最大值
        for (size_t j = 0; j < T; ++j) {
          size_t pos = i * T + j;
          if (att_ptr[pos] > maxval) {
            maxval = att_ptr[pos];
          }
        }
        float sum = 0.0f;
        for (size_t j = 0; j < T; ++j) {
          size_t pos = i * T + j;
          att_ptr[pos] = std::expf(att_ptr[pos] - maxval);
          sum += att_ptr[pos];
        }
        for (size_t j = 0; j < T; ++j) {
          size_t pos = i * T + j;
          att_ptr[pos] /= sum;
        }
      }
      // 在推理场景下，dropout不生效
      auto att_out = attn_dropout_->forward(att);
      auto att_out_ptr = reinterpret_cast<float *>(att_out.data());

      // 计算 attention @
      // V，然后再直接赋值到输出的指定位置，这里也包含将多头合并到最终的输出
      //  att_out[T,T] , v [T,head_dim_]
      for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < head_dim_; ++j) {
          float sum_a = 0.0f;
          for (size_t k = 0; k < T; ++k) {
            float *v_ptr =
                qkv_ptr + b * T * C3 + k * C3 + 2 * C + h * head_dim_;
            sum_a += att_out_ptr[i * T + k] * v_ptr[j];
          }
          y_ptr[b * T * C + i * C + h * head_dim_ + j] = sum_a;
        }
      }
    }
  }
  // 投影
  y = c_proj_->forward(y);
  // 推理这个dropout依然不起作用
  y = resid_dropout_->forward(y);
  return y;
}

// 如果使能kv cache，在这里处理
// 一般来说，对于自回归的模型第二次输出只有一个token,所以，第二次进入，T=1
dense::Tensor CausalSelfAttention::forward_cache(const dense::Tensor &input) {
  auto B = input.size(0); // 批大小
  auto T = input.size(1); // 序列长度
  auto C = input.size(2); // 嵌入维度

  // input首先分别于Q,K,V相乘，生成 Q,K,V三个矩阵
  // 因为Q,K,V三个矩阵在合在一起的，我们做一次乘法就全部完成了
  // qkv [B,T,C*3]
  auto qkv = c_attn_->forward(input);
  auto q = dense::Tensor::blank(qkv.dtype(), {B, T, C});
  auto k = dense::Tensor::blank(qkv.dtype(), {B, T, C});
  auto v = dense::Tensor::blank(qkv.dtype(), {B, T, C});
  {
    // 分离为三个独立张量，这里没有转置操作
    size_t C3 = C * 3;
    size_t B3 = T * C3; // qkv，每个批次的偏移量
    size_t B1 = T * C;  // q,k,v，每个张量的批次偏移量
    // 每个特征向量包含的数据长度，字节单位
    size_t CES = dense::get_element_size(qkv.dtype()) * C;

    float *qkv_ptr = reinterpret_cast<float *>(qkv.data());
    float *q_ptr = reinterpret_cast<float *>(q.data());
    float *k_ptr = reinterpret_cast<float *>(k.data());
    float *v_ptr = reinterpret_cast<float *>(v.data());

    for (size_t b_idx = 0; b_idx < B; ++b_idx) {
      auto qkv_base = qkv_ptr + b_idx * B3;
      auto q_base = q_ptr + b_idx * B1;
      auto k_base = k_ptr + b_idx * B1;
      auto v_base = v_ptr + b_idx * B1;

      for (size_t t = 0; t < T; ++t) {
        // 逐个token复制，特征长度都是相同的
        std::memcpy(q_base + t * C, qkv_base + t * C3, CES);
        std::memcpy(k_base + t * C, qkv_base + t * C3 + C, CES);
        std::memcpy(v_base + t * C, qkv_base + t * C3 + C * 2, CES);
      }
    }
  }

  // 找到这个层的kv cache对象
  auto cache = ctx_->cache_->get(index_);

  // 将新的token的k，v张量合并到cache中，形成一个完整的张量
  // 对于第二次，T=1，合并之后，cache中的张量就包含所有的token，但是之前的我们不用重复计算
  // 所以在cache下，推理的速度保持不变，不会因为token的增加导致计算变慢
  cache->update(k, v);

  // 我们取出包含全部token的 k，v 张量
  // 但是，查询(Q)的张量依然只有一个token（如果是第二次的话）
  k = cache->key_states();   // [B,total_seq_len,C]
  v = cache->value_states(); // [B,total_seq_len,C]

  int64_t total_seq_len =
      k.size(1); // 总的token长度，随着推理的进行，这个值每次加[1]

  float scale = 1.0 / sqrtf(head_dim_);

  // 每个头att的形状都是 [T,total_seq_len]
  // 这个形状是由 q*k.transpose() 决定,q[B,T,C],k的转置[B,C,total_seq_len] =>
  // [B,T,total_seq_len] 因为我们分批次计算，所以，att的形状不考虑批次 =>
  // [T,total_seq_len] 可以想象第一次推理，att的形状就是 [T,T]
  auto att = dense::Tensor::zeros(dense::kFloat32, {T, total_seq_len});
  auto att_ptr = reinterpret_cast<float *>(att.data());

  // 输出张量，这是合并了多头之后的输出形状，等同于输入形状
  auto y = dense::Tensor::zeros(dense::kFloat32, {B, T, C});
  auto y_ptr = reinterpret_cast<float *>(y.data());

  // 掩码张量，下三角
  auto mask_ptr = reinterpret_cast<int8_t *>(mask_.data());

  float *q_ptr = reinterpret_cast<float *>(q.data());
  float *k_ptr = reinterpret_cast<float *>(k.data());
  float *v_ptr = reinterpret_cast<float *>(v.data());

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {
      // 1. q*k.transpose()
      for (size_t i = 0; i < T; ++i) {
        float *q_base = q_ptr + b * T * C + i * C + h * head_dim_;
        for (size_t j = 0; j < total_seq_len; ++j) {
          float *k_base = k_ptr + b * total_seq_len * C + j * C + h * head_dim_;
          float val = 0.0f;
          for (size_t k = 0; k < head_dim_; ++k) {
            val += q_base[k] * k_base[k];
          }
          att_ptr[i * total_seq_len + j] = val * scale;
        }
      }

      // mask fill, att[T,total_seq_len]
      // 如果使用kv cache，我们只要对最后一行掩码填充，之前的都已经填充过了
      // 1
      // 1 1
      // 1 1 1
      // ...
      // 1 1 1 ....total_seq_len
      for (size_t i = total_seq_len - T; i < total_seq_len; ++i) {
        for (size_t j = 0; j < total_seq_len; ++j) {
          if (mask_ptr[i * config_.context_length + j] == 0) {
            att_ptr[i * total_seq_len + j] = -INFINITY;
          }
        }
      }

      // softmax, att[T,total_seq_len]
      for (size_t i = 0; i < T; ++i) {
        float maxval = -INFINITY;
        for (size_t j = 0; j < total_seq_len; ++j) {
          size_t pos = i * total_seq_len + j;
          if (att_ptr[pos] > maxval) {
            maxval = att_ptr[pos];
          }
        }

        float sum = 0.0f;
        for (size_t j = 0; j < total_seq_len; ++j) {
          size_t pos = i * total_seq_len + j;
          att_ptr[pos] = std::expf(att_ptr[pos] - maxval);
          sum += att_ptr[pos];
        }
        for (size_t j = 0; j < total_seq_len; ++j) {
          att_ptr[i * total_seq_len + j] /= sum;
        }
      }

      auto att_out = attn_dropout_->forward(att);
      auto att_out_ptr = reinterpret_cast<float *>(att_out.data());

      // 计算 attention @ V,然后合并到最终的输出Tensor中
      // att_out:[T,total_seq_len], v[B,total_seq_len,C]
      auto v_b_ptr = v_ptr + b * total_seq_len * C;

      for (size_t i = 0; i < T; ++i) {
        for (size_t j = 0; j < head_dim_; ++j) {
          float sum_a = 0.0f;
          for (size_t k = 0; k < total_seq_len; ++k) {
            float *v_base = v_b_ptr + k * C + h * head_dim_;
            sum_a += att_out_ptr[i * total_seq_len + k] * v_base[j];
          }
          y_ptr[b * T * C + i * C + h * head_dim_ + j] = sum_a;
        }
      }
    }
  }
  y = c_proj_->forward(y);
  y = resid_dropout_->forward(y);
  return y;
}

dense::Tensor CausalSelfAttention::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

Block::Block(Context *ctx, const std::string &instance_name,
             const ModelConfig &config, size_t idx)
    : Layer(ctx, instance_name), config_(config) {
  ln_1_ = std::make_unique<LayerNorm>(ctx, "ln_1", config_.emb_dim,
                                      config_.qkv_bias);
  attn_ = std::make_unique<CausalSelfAttention>(ctx, "attn", config_, idx);
  ln_2_ = std::make_unique<LayerNorm>(ctx, "ln_2", config_.emb_dim,
                                      config_.qkv_bias);
  mlp_ = std::make_unique<MLP>(ctx, "mlp", config_);
}

Block::~Block() = default;

dense::Tensor Block::forward(const dense::Tensor &input) {
  auto B = input.size(0); // 批大小
  auto T = input.size(1); // 序列长度
  auto C = input.size(2); // 嵌入维度

  auto N = B * T * C;

  shortcut_1_cache_ = input.clone();
  auto x = ln_1_->forward(input);
  x = attn_->forward(x);
  {
    // 第一个残差连接
    auto s_ptr = reinterpret_cast<float *>(shortcut_1_cache_.data());
    auto x_ptr = reinterpret_cast<float *>(x.data());
    for (size_t i = 0; i < N; ++i) {
      x_ptr[i] += s_ptr[i];
    }
  }
  shortcut_2_cache_ = x.clone();
  x = ln_2_->forward(x);
  x = mlp_->forward(x);
  {
    // 第二个残差连接
    auto s_ptr = reinterpret_cast<float *>(shortcut_2_cache_.data());
    auto x_ptr = reinterpret_cast<float *>(x.data());
    for (size_t i = 0; i < N; ++i) {
      x_ptr[i] += s_ptr[i];
    }
  }
  return x;
}

dense::Tensor Block::backward(const dense::Tensor &grad_output) {
  return dense::Tensor();
}

GPT::GPT(const ModelConfig &config) : config_(config), ctx_(new Context()) {
  wte_ = std::make_unique<Embedding>(ctx_.get(), "wte", config_.vocab_size,
                                     config_.emb_dim, 50256);
  wpe_ = std::make_unique<Embedding>(ctx_.get(), "wpe", config_.context_length,
                                     config_.emb_dim);
  dropout_ = std::make_unique<Dropout>(ctx_.get(), "dropout",
                                       config_.drop_rate /*0.1*/);
  // 创建 n_layers 个 Block 实例
  for (size_t i = 0; i < config_.n_layers; ++i) {
    auto block = std::make_unique<Block>(ctx_.get(), "h", config_, i);
    h_.emplace_back(std::move(block));
  }
  ln_f_ =
      std::make_unique<LayerNorm>(ctx_.get(), "ln_f", config_.emb_dim, true);
  lm_head_ = std::make_unique<Linear>(ctx_.get(), "lm_head", config_.emb_dim,
                                      config_.vocab_size, false);
}

GPT::~GPT() = default;

void GPT::from_pretrained(const std::string &filename) {
  if (!filename.empty()) {
    if (!dense::ModelParams::load(filename, &model_params_)) {
      return;
    }
    _load_weights();
  }
}

void GPT::save(const std::string &filename) {
  dense::ModelParams model_params;
  model_params.meta_data = model_params_.meta_data;
  size_t total_size = 0;
  total_size += _write_tensor(model_params, "wte.weight", wte_->W_);
  total_size += _write_tensor(model_params, "wpe.weight", wpe_->W_);
  total_size += _write_tensor(model_params, "ln_f.weight", ln_f_->W_);
  total_size += _write_tensor(model_params, "ln_f.bias", ln_f_->b_);
  for (size_t i = 0; i < h_.size(); ++i) {
    auto &block = h_[i];
    std::string prefix = "h." + std::to_string(i) + ".";
    total_size +=
        _write_tensor(model_params, prefix + "ln_1.weight", block->ln_1_->W_);
    total_size +=
        _write_tensor(model_params, prefix + "ln_1.bias", block->ln_1_->b_);
    total_size +=
        _write_tensor(model_params, prefix + "attn.c_attn.weight",
                      block->attn_->c_attn_->W_.clone().transpose_2d());
    total_size += _write_tensor(model_params, prefix + "attn.c_attn.bias",
                                block->attn_->c_attn_->b_);
    total_size +=
        _write_tensor(model_params, prefix + "attn.c_proj.weight",
                      block->attn_->c_proj_->W_.clone().transpose_2d());
    total_size += _write_tensor(model_params, prefix + "attn.c_proj.bias",
                                block->attn_->c_proj_->b_);
    total_size +=
        _write_tensor(model_params, prefix + "ln_2.weight", block->ln_2_->W_);
    total_size +=
        _write_tensor(model_params, prefix + "ln_2.bias", block->ln_2_->b_);
    total_size += _write_tensor(model_params, prefix + "mlp.c_fc.weight",
                                block->mlp_->c_fc_->W_.clone().transpose_2d());
    total_size += _write_tensor(model_params, prefix + "mlp.c_fc.bias",
                                block->mlp_->c_fc_->b_);
    total_size +=
        _write_tensor(model_params, prefix + "mlp.c_proj.weight",
                      block->mlp_->c_proj_->W_.clone().transpose_2d());
    total_size += _write_tensor(model_params, prefix + "mlp.c_proj.bias",
                                block->mlp_->c_proj_->b_);
  }
  std::cout << "模型参数总大小: " << total_size << " bytes" << std::endl;
  model_params.save(filename);
}

void GPT::enable_cache() {
  if (!ctx_->cache_) {
    ctx_->cache_ = std::make_unique<DynamicCache>(config_.n_layers,
                                                  config_.context_length);
  }
}

void GPT::enable_training(bool enable) { ctx_->training = enable; }

bool GPT::is_enable_cache() const {
  return ctx_->cache_ != nullptr && !ctx_->training;
}

dense::Tensor GPT::forward(const dense::Tensor &input) {
  auto B = input.size(0);
  auto T = input.size(1);
  assert(T <= config_.context_length);

  std::vector<int32_t> pos_data(B * T);
  size_t current_pos = 0;
  if (is_enable_cache()) {
    current_pos = ctx_->cache_->get_seq_length();
  }
  for (size_t i = 0; i < B; ++i) {
    for (size_t j = 0; j < T; ++j) {
      pos_data[i * T + j] = current_pos + static_cast<int32_t>(j);
    }
  }

  auto pos = dense::Tensor::from_blob(dense::kInt32, {B, T}, &pos_data[0]);

  auto tok_emb = wte_->forward(input);
  auto pos_emb = wpe_->forward(pos);
  auto tok_pos_emb = dense::Tensor::zeros(dense::kFloat32, tok_emb.shape());

  auto C = tok_pos_emb.size(-1);
  {
    auto tok_ptr = reinterpret_cast<float *>(tok_emb.data());
    auto pos_ptr = reinterpret_cast<float *>(pos_emb.data());
    auto out = reinterpret_cast<float *>(tok_pos_emb.data());
    for (size_t i = 0; i < B; ++i) {
      for (size_t j = 0; j < T; ++j) {
        for (size_t k = 0; k < C; ++k) {
          size_t idx = i * T * C + j * C + k;
          out[idx] = tok_ptr[idx] + pos_ptr[idx];
        }
      }
    }
  }

  auto x = dropout_->forward(tok_pos_emb);

  for (size_t i = 0; i < h_.size(); ++i) {
    auto &block = h_[i];
    x = block->forward(x);
  }
  x = ln_f_->forward(x);
  auto logits = lm_head_->forward(x);
  return logits;
}

std::vector<int> GPT::inference(std::vector<int> tokens, int max_length,
                                SamplingChain *chain,
                                std::function<bool(int)> token_callback) {
  assert(chain != nullptr);
  if (is_enable_cache() &&
      (max_length + tokens.size() > config_.context_length)) {
    std::cerr << "错误:启用缓存时,生成长度加上初始token数量不能超过模型上下文"
                 "长度。\n";
    return {};
  }

  std::vector<int> result_tokens = tokens;
  while (result_tokens.size() < max_length) {
    dense::Tensor input_tensor;
    if (is_enable_cache()) {
      if (result_tokens.size() == tokens.size()) {
        input_tensor = dense::Tensor::from_blob(
            dense::kInt32, {1, static_cast<int64_t>(result_tokens.size())},
            &result_tokens[0]);
      } else {
        input_tensor = dense::Tensor::from_blob(
            dense::kInt32, {1, 1}, &result_tokens[result_tokens.size() - 1]);
      }
    } else {
      input_tensor = dense::Tensor::from_blob(
          dense::kInt32, {1, static_cast<int64_t>(result_tokens.size())},
          &result_tokens[0]);
    }
    auto logits = forward(input_tensor);
    //  【B,T,C]
    auto B = logits.size(0);
    auto T = logits.size(1);
    auto C = logits.size(2);
    // 只取最后一个token预测向量
    auto offset = (T - 1) * C * sizeof(float);
    auto ptr = reinterpret_cast<float *>(logits.data() + offset);
    auto logits_tensor = dense::Tensor::from_blob(logits.dtype(), {1, C}, ptr);

    auto logits_ptr = reinterpret_cast<float *>(logits_tensor.data());

    std::vector<llama_token_data> cur;
    cur.reserve(logits_tensor.numel());
    for (llama_token token_id = 0; token_id < logits_tensor.numel();
         ++token_id) {
      cur.emplace_back(llama_token_data(token_id, logits_ptr[token_id], 0.0f));
    }

    llama_token_data_array cur_p(&cur[0], cur.size());
    auto next_token_id = chain->sample(&cur_p);

    result_tokens.push_back(next_token_id);

    if (token_callback) {
      // 如果提供了回调函数，调用它来处理生成的token。
      if (!token_callback(next_token_id)) {
        break;
      }
    } else if (next_token_id == 50256) { // GPT-2的EOS token ID通常是50256
      // std::cerr << "\n生成了EOS token，提前停止生成。\n";
      break;
    }
  }

  return result_tokens;
}

dense::Tensor CreateTensor(const dense::TensorInfo &info) {
  auto tensor = dense::Tensor::from_blob(dense::dtype_from_string(info.dtype),
                                         info.shape, info.data_ptr);
  return tensor;
}

void GPT::_load_weights() {
  auto params = model_params_.tensors;
  auto wte_weight = CreateTensor(params.at("wte.weight"));

  wte_->W_ = wte_weight;
  lm_head_->W_ = wte_weight;

  wpe_->W_ = CreateTensor(params.at("wpe.weight"));

  ln_f_->W_ = CreateTensor(params.at("ln_f.weight"));
  ln_f_->b_ = CreateTensor(params.at("ln_f.bias"));

  for (size_t i = 0; i < h_.size(); ++i) {
    auto &block = h_[i];
    std::string prefix = "h." + std::to_string(i) + ".";

    block->ln_1_->W_ = CreateTensor(params.at(prefix + "ln_1.weight"));
    block->ln_1_->b_ = CreateTensor(params.at(prefix + "ln_1.bias"));

    // 这里要转置
    block->attn_->c_attn_->W_ =
        CreateTensor(params.at(prefix + "attn.c_attn.weight")).transpose_2d();

    block->attn_->c_attn_->b_ =
        CreateTensor(params.at(prefix + "attn.c_attn.bias"));

    // 这里要转置
    block->attn_->c_proj_->W_ =
        CreateTensor(params.at(prefix + "attn.c_proj.weight")).transpose_2d();

    block->attn_->c_proj_->b_ =
        CreateTensor(params.at(prefix + "attn.c_proj.bias"));

    block->ln_2_->W_ = CreateTensor(params.at(prefix + "ln_2.weight"));
    block->ln_2_->b_ = CreateTensor(params.at(prefix + "ln_2.bias"));
    // 这里要转置
    block->mlp_->c_fc_->W_ =
        CreateTensor(params.at(prefix + "mlp.c_fc.weight")).transpose_2d();

    block->mlp_->c_fc_->b_ = CreateTensor(params.at(prefix + "mlp.c_fc.bias"));
    // 这里要转置
    block->mlp_->c_proj_->W_ =
        CreateTensor(params.at(prefix + "mlp.c_proj.weight")).transpose_2d();

    block->mlp_->c_proj_->b_ =
        CreateTensor(params.at(prefix + "mlp.c_proj.bias"));
  }
}

size_t GPT::_write_tensor(dense::ModelParams &model_params,
                          const std::string &name,
                          const dense::Tensor &tensor) {
  if (tensor.numel() == 0) {
    std::cerr << "Warning: Tensor '" << name << "' is null, skipping write."
              << std::endl;
    return 0;
  }

  dense::TensorInfo info;
  info.storage = std::make_shared<dense::Storage>(tensor.data_size());
  info.dtype = dense::dtype_to_string(tensor.dtype());
  info.shape = tensor.shape();
  info.data_ptr = info.storage->data();
  info.data_size = tensor.data_size();
  std::memcpy(info.storage->data(), tensor.data(), tensor.data_size());
  model_params.tensors[name] = info;
  return info.data_size;
}
