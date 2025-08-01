#include "layers.h"
#include "cache.h"
#include "json.hpp"
#include "sampling.h"
#include <algorithm>
#include <fstream>
#include <functional>
#include <iosfwd>
#include <iostream>
#include <math.h>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>

namespace {

void normal_fill(float *data, size_t num_elements, double mean, double std) {
  std::mt19937 gen(std::random_device{}());
  std::normal_distribution<double> normal_dist(mean, std);
  for (size_t i = 0; i < num_elements; ++i) {
    data[i] = static_cast<float>(normal_dist(gen));
  }
}

std::vector<dense::Tensor> tensor_split(const dense::Tensor &A,
                                        int64_t split_size) {
  auto B = A.size(0);
  auto T = A.size(1);
  auto C = A.size(2);

  int64_t ncount = C / split_size;

  std::vector<dense::Tensor> result;
  for (size_t i = 0; i < ncount; ++i) {
    auto t = dense::Tensor::zeros(A.dtype(), {B, T, split_size});
    result.emplace_back(std::move(t));
  }

  // 每个特征向量包含的数据长度，字节单位
  size_t data_size = dense::get_element_size(A.dtype()) * split_size;

  auto A_ptr = reinterpret_cast<const float *>(A.data());

  for (size_t b = 0; b < B; ++b) {
    auto a_bt = A_ptr + b * T * C;

    for (size_t t = 0; t < T; ++t) {
      // 逐个token复制，特征长度都是相同的
      size_t data_offset = b * T * split_size + t * split_size;
      for (size_t i = 0; i < result.size(); ++i) {
        auto data_ptr = reinterpret_cast<float *>(result[i].data());
        std::memcpy(data_ptr + data_offset, a_bt + t * C + i * split_size,
                    data_size);
      }
    }
  }
  return result;
}

} // namespace

ModelConfig::ModelConfig()
    : vocab_size(50257), context_length(1024), emb_dim(768), n_heads(12),
      n_layers(12), drop_rate(0.1), qkv_bias(true), expansion_ratio(4.0f),
      ln_epsilon(1e-05) {}
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
    if (config_json.contains("layer_norm_epsilon")) {
      ln_epsilon = config_json.at("layer_norm_epsilon").get<float>();
    }
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
  // 嵌入层有可学习参数 W_（嵌入矩阵）
  ctx_->Register(this);
  // WTE[50276,768],WPE[1024,768]
  W_ = dense::Tensor(dense::kFloat32, {num_embeddings_, embedding_dim_});
}

Embedding::~Embedding() = default;

dense::Tensor Embedding::forward(const dense::Tensor &input) {
  if (input.dim() != 2) {
    throw std::runtime_error("输入张量的形状必须是[B,T]");
  }
  if (input.dtype() != dense::DType::kInt32) {
    throw std::runtime_error("输入张量的类型必须是:kInt32");
  }

  // input 的形状限定为 [B,T]
  // 输出形状为 [B,T,C]
  // WTE is (vocab_size,C) ->[50257,768],token嵌入矩阵
  // WPE is (max_token,C)->[1024,768],位置嵌入矩阵

  if (is_training()) {
    input_cache_ = input.clone();
  }

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = W_.size(-1);

  auto output = dense::Tensor::zeros(dense::kFloat32, {B, T, C});

  auto out_ptr = reinterpret_cast<float *>(output.data());
  auto in_ptr = reinterpret_cast<const int32_t *>(input.data());
  auto w_ptr = reinterpret_cast<float *>(W_.data());

  // 如果 input[0,0] = 10
  // 将嵌入矩阵的行索引=10的那个特征向量(C)复制给输出的[0,0]所在行
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      auto out_bt = out_ptr + b * T * C + t * C; // output[b,t,:]
      auto idx = in_ptr[b * T + t];              // input[b, t]
      // idx 是嵌入索引
      // 将 w_ptr 的指针移动到 idx 索引对应的行开始位置
      auto w_idx = w_ptr + idx * C;
      // 将嵌入矩阵的这个特征向量复制给输出
      for (size_t i = 0; i < C; ++i) {
        // 这里可用 std::memcpy替代，我们只是展示详细的计算过程
        out_bt[i] = w_idx[i];
      }
    }
  }
  return output;
}

dense::Tensor Embedding::backward(const dense::Tensor &grad_output) {
  // grad_output形状: [B,T,C],比如这里[1,7,768]
  // 计算的核心过程是将输入的索引映射到嵌入矩阵的行上，并累加梯度
  // 比如 T[t]=1,则需要将梯度累加到 W_[1,:] 上

  if (!grad_W_.is_defined()) {
    grad_W_ = dense::Tensor::zeros_like(W_);
  }

  auto input = input_cache_; //[B,T]

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = W_.size(-1); // 嵌入维度，比如768

  auto in_ptr = reinterpret_cast<const int32_t *>(input.data());
  auto grad_out_ptr = reinterpret_cast<const float *>(grad_output.data());
  auto grad_w_ptr = reinterpret_cast<float *>(grad_W_.data());

  // grad_W_形状: [50257,768]或者[1024,768],这里是累加梯度
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      auto idx = in_ptr[b * T + t]; // 获取输入的索引
      if (idx < 0 || idx >= num_embeddings_) {
        continue; // 忽略无效索引
      }
      auto grad_out_bt = grad_out_ptr + b * T * C + t * C;
      auto grad_w_idx = grad_w_ptr + idx * C;
      for (size_t i = 0; i < C; ++i) {
        grad_w_idx[i] += grad_out_bt[i]; // 累加梯度
      }
    }
  }
  // 返回空张量，因为嵌入层没有输入梯度
  // 在实际应用中，嵌入层通常不需要返回输入梯度
  return dense::Tensor();
}

Linear::Linear(Context *ctx, const std::string &instance_name,
               int64_t in_features, int64_t out_features, bool has_bias)
    : Layer(ctx, instance_name), in_features_(in_features),
      out_features_(out_features), has_bias_(has_bias) {
  // 线性层有可学习参数 W_,b_(if has_bias_=true)
  ctx_->Register(this);
  // torch::nn::Linear 的权重形状是 [out_features, in_features]
  W_ = dense::Tensor(dense::kFloat32, {out_features_, in_features_});

  if (has_bias) {
    b_ = dense::Tensor(dense::kFloat32, {out_features_});
  }
}

Linear::~Linear() = default;

dense::Tensor Linear::forward(const dense::Tensor &input) {
  // 这里输入 input的形状是 [B, T,D_in]
  // 输出 output 的形状是 [B, T,D_out]

  auto B = input.size(0);
  auto T = input.size(1);
  auto D_in = input.size(2);
  auto D_out = W_.size(0); //[D_out, D_in]

  if (is_training()) {
    // 反向传播计算要用到
    input_cache_ = input.clone();
  }

  // output 必须清零，matmul_B_transpose 实现的是累积，不是直接赋值
  auto output = dense::Tensor::zeros(input.dtype(), {B, T, D_out});

  auto in_ptr = reinterpret_cast<const float *>(input.data());
  auto w_ptr = reinterpret_cast<float *>(W_.data());
  auto b_ptr = reinterpret_cast<float *>(b_.data());
  auto out_ptr = reinterpret_cast<float *>(output.data());

  // Y = X @ W^T + b
  for (size_t b = 0; b < B; ++b) {
    // 每个批次都执行相同的矩阵乘法
    auto inp_bt = in_ptr + b * T * D_in;    // 输入的第 b 个批次
    auto outp_bt = out_ptr + b * T * D_out; // 输出的第 b 个批次
    // inp_bt[T,C],outp_bt[T,OC]
    dense::matmul_B_transpose(inp_bt, D_in, w_ptr, D_in,
                              has_bias_ ? b_ptr : nullptr, outp_bt, D_out,
                              T /*M*/, D_out /*N*/, D_in /*K*/);
  }
  return output;
}

dense::Tensor Linear::backward(const dense::Tensor &grad_output) {
  // grad_output 的形状: [B,T,D_out] (来自上游的损失对 Y 的梯度)
  // X (input_cache_) 的形状: [B,T, D_in] (前向时的输入)
  // W_ 的形状: [D_out, D_in] (权重矩阵)
  // b_ 的形状: [D_out] (偏置向量)

  if (!grad_W_.is_defined()) {
    // grad_W_ 的形状: [D_out, D_in] (权重梯度矩阵)
    grad_W_ = dense::Tensor::zeros_like(W_);
  }
  if (!grad_b_.is_defined()) {
    // grad_b_ 的形状: [D_out] (偏置梯度向量)
    grad_b_ = dense::Tensor::zeros_like(b_);
  }

  auto input = input_cache_; // 前向时的输入

  auto B = input.size(0);            // 批次大小
  auto T = input.size(1);            // 序列长度
  auto D_in = input.size(-1);        // 输入维度
  auto D_out = grad_output.size(-1); // 输出维度

  auto grad_out_ptr = reinterpret_cast<const float *>(grad_output.data());
  auto in_ptr = reinterpret_cast<const float *>(input.data());
  auto grad_w_ptr = reinterpret_cast<float *>(grad_W_.data());
  auto w_ptr = reinterpret_cast<float *>(W_.data());

  // grad_input 的形状: [B,T,D_in]
  // 返回给上一层的梯度，形状应当与前向传播输入(input)一致
  auto grad_input = dense::Tensor::zeros(dense::kFloat32, {B, T, D_in});
  auto grad_in_ptr = reinterpret_cast<float *>(grad_input.data());

  // 计算对 W_ 的梯度，这里包含转置乘法，公式是: grad_W = grad_output^T @ X
  for (size_t b = 0; b < B; ++b) {
    auto grad_outp_bt = grad_out_ptr + b * T * D_out; // 当前批次的梯度
    auto in_bt = in_ptr + b * T * D_in;               // 当前批次的输入
    dense::matmul_A_transpose(grad_outp_bt, D_out, in_bt, D_in, nullptr,
                              grad_w_ptr, D_in, T /*K*/, D_out /*M*/,
                              D_in /*N*/);
  }
  if (has_bias_) {
    /* 计算对 b_ 的梯度，公式是: grad_b = sum(grad_output, axis=0)
       这里我们只需要对输出维度进行累加，假如 grad_output 形状[1,3,4]
       [
        [1,2,3,4]
        [5,6,7,8]
        [9,10,11,12]
       ]
       grad_b[0] = 1+5+9  = 15
       grad_b[1] = 2+6+10 = 18
       grad_b[2] = 3+7+11 = 21
       grad_b[3] = 4+8+12 = 24
    */
    auto grad_bp = reinterpret_cast<float *>(grad_b_.data());
    for (size_t i = 0; i < D_out; ++i) {
      float sum = 0.0f;
      for (size_t j = 0; j < B * T; ++j) {
        sum += grad_out_ptr[j * D_out + i];
      }
      grad_bp[i] += sum;
    }
  }

  // 对 输入 X 的梯度计算
  // 公式是: grad_X = grad_output @ W_
  for (size_t b = 0; b < B; ++b) {
    auto grad_outp_bt = grad_out_ptr + b * T * D_out;
    auto grad_inp_bt = grad_in_ptr + b * T * D_in;
    // grad_outp_bt[T,D_out], w_ptr[D_out,D_in]
    dense::matmul(grad_outp_bt, D_out, w_ptr, D_in, nullptr, grad_inp_bt, D_in,
                  T /*M*/, D_out /*K*/, D_in /*N*/);
  }
  return grad_input;
}

LayerNorm::LayerNorm(Context *ctx, const std::string &instance_name,
                     int64_t ndim, float epsilon, bool has_bias)
    : Layer(ctx, instance_name), ndim_(ndim), epsilon_(epsilon),
      has_bias_(has_bias) {
  // 归一化层有可学习参数 W_(gamma),b_(beta)(if has_bias=true)
  ctx_->Register(this);
  // ndim_ 是归一化的维度大小,等同于嵌入维度，GPT2 中通常是 768
  W_ = dense::Tensor(dense::kFloat32, {ndim_});
  if (has_bias) {
    b_ = dense::Tensor(dense::kFloat32, {ndim_});
  }
}

LayerNorm::~LayerNorm() = default;

dense::Tensor LayerNorm::forward(const dense::Tensor &input) {
  // 输入input的形状是 [B, T, ndim_]
  // ndim_ 是归一化的维度大小,在 GPT-2 中通常是 768
  // W_ 的形状是 [ndim_], b_ 的形状也是 [ndim_]
  // 输出 output 的形状与输入相同，即 [B, T, ndim_]
  auto B = input.size(0);
  auto T = input.size(1);
  auto C = input.size(2);

  auto output = dense::Tensor::zeros_like(input);

  float *mean_ptr = nullptr;
  float *rstd_ptr = nullptr;

  if (is_training()) {
    // 反向传播所需，原始输入以及一些中间计算结果
    input_cache_ = input.clone();
    mean_ = dense::Tensor::zeros(input.dtype(), {B, T});
    rstd_ = dense::Tensor::zeros(input.dtype(), {B, T});
    mean_ptr = reinterpret_cast<float *>(mean_.data());
    rstd_ptr = reinterpret_cast<float *>(rstd_.data());
  }

  auto in_ptr = reinterpret_cast<const float *>(input.data());
  auto out_ptr = reinterpret_cast<float *>(output.data());
  auto w_ptr = reinterpret_cast<float *>(W_.data()); // gamma
  auto b_ptr = reinterpret_cast<float *>(b_.data()); // beta

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {

      auto in_bt = in_ptr + b * T * C + t * C;
      auto out_bt = out_ptr + b * T * C + t * C;

      float mean = 0.0f;
      for (size_t i = 0; i < C; ++i) {
        mean += in_bt[i];
      }
      mean = mean / C; // 均值

      float var = 0.0f;
      for (size_t i = 0; i < C; ++i) {
        float xshift = in_bt[i] - mean;
        var += xshift * xshift;
      }
      var = var / C; // 方差

      float rstd = 1.0f / std::sqrt(var + epsilon_);

      for (size_t i = 0; i < C; ++i) {
        // x_hat = (x-mean)/sqrt(var+eplison)
        float x_hat = (rstd * (in_bt[i] - mean));
        // y = x_hat*gamma + beta
        out_bt[i] = x_hat * w_ptr[i];
        if (has_bias_) {
          // bias 不是必须的
          out_bt[i] += b_ptr[i];
        }
      }

      if (is_training()) {
        mean_ptr[b * T + t] = mean; // backward 要用到，避免重复计算
        rstd_ptr[b * T + t] = rstd; // backward 要用到，避免重复计算
      }
    }
  }
  return output;
}

dense::Tensor LayerNorm::backward(const dense::Tensor &grad_output) {
  // grad_output 形状: [B, T, ndim_], 例如 [1, 7, 768]
  // W_ (gamma) 形状: [ndim_]
  // b_ (beta) 形状: [ndim_]

  if (!grad_W_.is_defined()) {
    grad_W_ = dense::Tensor::zeros_like(W_);
  }
  if (has_bias_ && !grad_b_.is_defined()) {
    grad_b_ = dense::Tensor::zeros_like(b_);
  }

  auto input = input_cache_; // 前向时的输入
  auto B = input.size(0);    // 批次大小
  auto T = input.size(1);    // 序列长度
  auto C = input.size(2);    // 归一化维度大小

  auto grad_out_ptr = reinterpret_cast<const float *>(grad_output.data());
  auto in_ptr = reinterpret_cast<const float *>(input.data());
  auto grad_w_ptr = reinterpret_cast<float *>(grad_W_.data());
  auto w_ptr = reinterpret_cast<const float *>(W_.data());
  auto grad_b_ptr = reinterpret_cast<float *>(grad_b_.data());
  auto mean_ptr = reinterpret_cast<const float *>(mean_.data());
  auto rstd_ptr = reinterpret_cast<const float *>(rstd_.data());

  // grad_input 的形状: [B,T,C]
  // 返回给上一层的梯度，形状应当与 grad_output 一致
  auto grad_input = dense::Tensor::zeros_like(input);
  auto grad_in_ptr = reinterpret_cast<float *>(grad_input.data());

  std::unique_ptr<float[]> x_hat(new float[C]);
  std::unique_ptr<float[]> dl_dx_hat(new float[C]);

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      // 这三个形状相同
      auto in_bt = in_ptr + b * T * C + t * C;
      auto grad_out_bt = grad_out_ptr + b * T * C + t * C;
      auto grad_in_bt = grad_in_ptr + b * T * C + t * C;

      // 这两个形状相同，是二维张量，在前向传播过程中保存的中间值
      auto mean_bt = mean_ptr[b * T + t];
      auto rstd_bt = rstd_ptr[b * T + t];

      float dl_dx_hat_sum = 0.0f;
      float dl_dx_hat_dot_x_hat = 0.0f;

      for (size_t i = 0; i < C; ++i) {
        x_hat[i] = (in_bt[i] - mean_bt) * rstd_bt;
        // dL/dx_hat = dL/dy * gamma
        dl_dx_hat[i] = grad_out_bt[i] * w_ptr[i];
        // Σ(dL/dx_hat),最后计算的时候要用到的中间值
        dl_dx_hat_sum += dl_dx_hat[i];
        // Σ(x_hat * dL/dx_hat)
        dl_dx_hat_dot_x_hat += x_hat[i] * dl_dx_hat[i];

        // dL/db = Σ(dL/dy)
        grad_b_ptr[i] += grad_out_bt[i];
        // dL/dW = Σ(dL/dy * x_hat)
        grad_w_ptr[i] += grad_out_bt[i] * x_hat[i];
      }

      float rtsd_mean = (1.0 / C) * rstd_bt;

      for (size_t i = 0; i < C; ++i) {
        float in = rtsd_mean * (C * dl_dx_hat[i] - dl_dx_hat_sum -
                                x_hat[i] * dl_dx_hat_dot_x_hat);

        grad_in_bt[i] = in;
      }
    }
  }
  return grad_input;
}

GELU::GELU(Context *ctx, const std::string &instance_name)
    : Layer(ctx, instance_name) {
  // 这一层没有可学习参数
}

GELU::~GELU() = default;

#define GELU_SCALING_FACTOR std::sqrt(2.0f / std::_Pi_val)

dense::Tensor GELU::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [B, T, C]
  // 输出 output 的形状与输入相同，即 [B, T, C]
  // GELU 激活函数的公式是: x * P(X <= x) = 0.5 * x * (1 + tanh(sqrt(2 / π) * (x
  // + 0.044715 * x^3)))
  if (is_training()) {
    input_cache_ = input.clone();
  }

  auto B = input.size(0);
  auto T = input.size(1);
  auto C = input.size(2);

  auto N = input.numel();

  auto output = dense::Tensor::zeros_like(input);

  auto in_ptr = reinterpret_cast<const float *>(input.data());
  auto out_ptr = reinterpret_cast<float *>(output.data());

  for (size_t i = 0; i < N; ++i) {
    float x = in_ptr[i];
    float cube = 0.044715f * std::pow(x, 3);
    out_ptr[i] =
        0.5f * x * (1.0f + std::tanh(GELU_SCALING_FACTOR * (x + cube)));
  }
  return output;
}

dense::Tensor GELU::backward(const dense::Tensor &grad_output) {
  // 输入 input 的形状是 [B, T, C]
  // grad_output 的形状是 [B, T, C]

  // GELU 激活函数的导数计算：GELU'(x)
  // GELU'(x) = 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) +
  //             0.5 * x * (1 - tanh^2(sqrt(2/pi) * (x + 0.044715 * x^3))) *
  //             sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
  auto input = input_cache_;

  auto N = input.numel();

  auto grad_output_ptr = reinterpret_cast<const float *>(grad_output.data());
  auto in_ptr = reinterpret_cast<const float *>(input.data());

  auto grad_input = dense::Tensor::zeros_like(grad_output);
  auto grad_input_ptr = reinterpret_cast<float *>(grad_input.data());

  for (size_t i = 0; i < N; ++i) {
    float x = in_ptr[i];

    float u = GELU_SCALING_FACTOR * (x + 0.044715f * std::pow(x, 3));
    float du_dx =
        GELU_SCALING_FACTOR * (1.0 + 0.044715f * 3.0 * std::pow(x, 2));
    float tanh_val = std::tanh(u);

    auto dg_dx =
        0.5 * (1 + tanh_val) + 0.5 * x * (1 - std::pow(tanh_val, 2)) * du_dx;
    grad_input_ptr[i] = dg_dx * grad_output_ptr[i];
  }
  return grad_input;
}

Dropout::Dropout(Context *ctx, const std::string &instance_name,
                 float dropout_ratio)
    : Layer(ctx, instance_name), dropout_ratio_(dropout_ratio) {
  // 这一层没有可学习的参数
  if (dropout_ratio < 0.0 || dropout_ratio >= 1.0) {
    throw std::invalid_argument("dropout probability must be in [0, 1)");
  }
  scale_ = (dropout_ratio == 0.0) ? 1.0 : 1.0 / (1.0 - dropout_ratio);
}

Dropout::~Dropout() = default;

dense::Tensor Dropout::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [B, T, C] 或者 [T, C](在多头注意力计算时)
  // 输出 output 的形状与输入相同
  // 在训练模式下，随机丢弃一部分神经元

  if (!is_training())
    return input;

  auto C = input.size(-1);
  int64_t bt = input.numel() / C;

  auto in_ptr = reinterpret_cast<const float *>(input.data());
  mask_ = dense::Tensor::rand(input.dtype(), input.shape());
  auto mask_ptr = reinterpret_cast<float *>(mask_.data());
  auto output = dense::Tensor::zeros_like(input);
  auto out_ptr = reinterpret_cast<float *>(output.data());

  for (size_t b = 0; b < bt; ++b) {
    auto out_bt = out_ptr + b * C;
    auto mask_bt = mask_ptr + b * C;
    auto in_bt = in_ptr + b * C;

    for (size_t i = 0; i < C; ++i) {
      mask_bt[i] = mask_bt[i] > dropout_ratio_ ? 1.0f : 0.0f;
      out_bt[i] = in_bt[i] * mask_bt[i] * scale_;
    }
  }

  return output;
}

dense::Tensor Dropout::backward(const dense::Tensor &grad_output) {
  // 梯度也要通过相同的掩码和缩放因子
  auto C = grad_output.size(-1);
  int64_t bt = grad_output.numel() / C;

  auto grad_output_ptr = reinterpret_cast<const float *>(grad_output.data());
  auto mask_ptr = reinterpret_cast<const float *>(mask_.data());
  auto grad_input = dense::Tensor::zeros_like(grad_output);
  auto grad_input_ptr = reinterpret_cast<float *>(grad_input.data());

  for (size_t b = 0; b < bt; ++b) {
    auto grad_input_bt = grad_input_ptr + b * C;
    auto grad_output_bt = grad_output_ptr + b * C;
    auto mask_bt = mask_ptr + b * C;
    for (size_t i = 0; i < C; ++i) {
      grad_input_bt[i] = grad_output_bt[i] * mask_bt[i] * scale_;
    }
  }
  return grad_input;
}

MLP::MLP(Context *ctx, const std::string &instance_name,
         const ModelConfig &config)
    : Layer(ctx, instance_name), config_(config) {
  // config_.emb_dim 是嵌入维度,通常是 768
  auto out_dim =
      static_cast<int64_t>(config_.expansion_ratio * config_.emb_dim);
  c_fc_ = std::make_unique<Linear>(ctx, "c_fc", config_.emb_dim, out_dim);
  gelu_ = std::make_unique<GELU>(ctx, "gelu");
  c_proj_ = std::make_unique<Linear>(ctx, "c_proj", out_dim, config_.emb_dim);
  dropout_ = std::make_unique<Dropout>(ctx, "dropout", config_.drop_rate);
}

MLP::~MLP() = default;

dense::Tensor MLP::forward(const dense::Tensor &input) {
  // 输入 input 的形状是 [B ,T ,C]
  // 输出 output 的形状也是 [B ,T ,C]

  // 从低维空间向高维空间投影
  auto x = c_fc_->forward(input);
  // 激活
  x = gelu_->forward(x);
  // 再从高维向低维投影
  x = c_proj_->forward(x);
  // 经过上述两次投影，保证输入和输出形状不变

  // 随机丢弃一些神经元
  x = dropout_->forward(x);
  return x;
}

dense::Tensor MLP::backward(const dense::Tensor &grad_output) {
  auto grad_input = dropout_->backward(grad_output);
  grad_input = c_proj_->backward(grad_input);
  grad_input = gelu_->backward(grad_input);
  grad_input = c_fc_->backward(grad_input);
  return grad_input;
}

CausalSelfAttention::CausalSelfAttention(Context *ctx,
                                         const std::string &instance_name,
                                         const ModelConfig &config, size_t idx)
    : Layer(ctx, instance_name), config_(config), head_dim_(0), index_(idx),
      scale_(0.0f) {
  assert(config_.emb_dim % config_.n_heads == 0);

  // 每个头的维度
  head_dim_ = static_cast<int64_t>(config_.emb_dim / config_.n_heads);

  // 缩放因子
  scale_ = 1.0 / std::sqrt(head_dim_);

  // 用于计算 X 与 Q,K,V 三个权重矩阵乘法
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
    auto M = mask_.size(0);
    auto N = mask_.size(1);
    for (size_t m = 0; m < M; ++m) {
      for (size_t n = 0; n < N; ++n) {
        if (n > m)
          ptr[m * N + n] = 0;
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

  // input首先分别于Q,K,V相乘，生成 Q,K,V 三个矩阵
  // 因为Q,K,V三个矩阵在合在一起的，我们做一次乘法就全部完成了
  // cached_qkv_: [B,T,C*3]
  cached_qkv_ = c_attn_->forward(input);
  auto C3 = cached_qkv_.size(-1);

  // 每个头att的形状都是 [T,T],用于保存中间计算结果
  auto att = dense::Tensor::zeros(dense::kFloat32, {T, T});

  if (is_training()) {
    // 缓存 softmax 的输出，因为 softmax 反向传播时依赖前向输出
    cached_att_after_softmax_ =
        dense::Tensor::zeros(dense::kFloat32, {B, config_.n_heads, T, T});
    // 缓存 dropout 的输出
    cached_att_after_dropout_ =
        dense::Tensor::zeros(dense::kFloat32, {B, config_.n_heads, T, T});
  }

  // 输出张量
  auto output = dense::Tensor::zeros_like(input);

  auto qkv_ptr = reinterpret_cast<float *>(cached_qkv_.data());
  auto out_ptr = reinterpret_cast<float *>(output.data());

  auto q_ptr = qkv_ptr;
  auto k_ptr = qkv_ptr + C;
  auto v_ptr = qkv_ptr + C * 2;

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {
      header_forward(q_ptr, C3, k_ptr, C3, v_ptr, C3, out_ptr, C, att, b, h);
    }
  }
  // output:[B,T,C]
  output = c_proj_->forward(output);
  output = resid_dropout_->forward(output);
  return output;
}

void CausalSelfAttention::header_forward(float *q, size_t q_stride, float *k,
                                         size_t k_stride, float *v,
                                         size_t v_stride, float *out,
                                         size_t out_stride, dense::Tensor &att,
                                         size_t b, size_t h) {

  auto T = att.size(0);
  auto total_seq_len = att.size(1);
  auto C = out_stride;

  auto q_bt = q + b * T * q_stride;
  auto k_bt = k + b * total_seq_len * k_stride;
  auto v_bt = v + b * total_seq_len * v_stride;

  att.zero_(); // att 清零，矩阵乘法输出是累加，不是赋值
  auto att_ptr = reinterpret_cast<float *>(att.data());

  auto out_bt = out + b * T * C; // 当前批次的输出起始位置
  auto mask_ptr = reinterpret_cast<const int8_t *>(mask_.data());

  // 1. Q*K^T, [T,head_dim_] @ [total_seq_len,head_dim_]^T --> [T,total_seq_len]
  dense::matmul_B_transpose(q_bt + h * head_dim_, // Q 的起始位置
                            q_stride,             // Q 的行距
                            k_bt + h * head_dim_, // K 的起始位置
                            k_stride,             // K 的行距
                            nullptr,              // bias
                            att_ptr,              // att:输出结果
                            total_seq_len,        // att 的行距
                            T,                    // M
                            total_seq_len,        // N
                            head_dim_             // K
  );

  // 对 att 逐元素缩放
  for (size_t i = 0; i < att.numel(); ++i) {
    att_ptr[i] *= scale_; // 缩放
  }

  // 对 att[T,total_seq_len] 应用掩码
  // 掩码矩阵是 [config_.context_length,config_.context_length]
  for (size_t i = total_seq_len - T; i < total_seq_len; ++i) {
    for (size_t j = 0; j < total_seq_len; ++j) {
      if (mask_ptr[i * config_.context_length + j] == 0) {
        att_ptr[i * total_seq_len + j] = -INFINITY;
      }
    }
  }

  // 计算 att 的 softmax
  dense::mat_softmax_forward(att_ptr, T, total_seq_len);

  if (is_training()) {
    // 缓存 softmax结算结果，在 softmax backward时需要用到
    auto ptr = reinterpret_cast<float *>(cached_att_after_softmax_.data()) +
               b * config_.n_heads * T * total_seq_len +
               h * T * total_seq_len;           // 每个批次每个头的起始位置
    std::memcpy(ptr, att_ptr, att.data_size()); // 缓存softmax的注意力
  }

  // 在推理场景下，dropout 不生效
  // drop_output[T,total_seq_len]
  auto drop_output = attn_dropout_->forward(att);
  auto drop_output_ptr = reinterpret_cast<float *>(drop_output.data());

  if (is_training()) {
    auto ptr = reinterpret_cast<float *>(cached_att_after_dropout_.data()) +
               b * config_.n_heads * T * total_seq_len +
               h * T * total_seq_len; // 每个批次每个头的起始位置
    std::memcpy(ptr, drop_output_ptr,
                drop_output.data_size()); // 缓存 dropout 后的 attention
  }

  // 计算 attention @ V
  // 然后再直接赋值到输出的指定位置，这里也包含将多头合并到最终的输出
  dense::matmul(drop_output_ptr, total_seq_len, v_bt + h * head_dim_, v_stride,
                nullptr, out_bt + h * head_dim_, C, T /*M*/,
                total_seq_len /*K*/, head_dim_ /*N*/);
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

  auto qkv_chunks = tensor_split(qkv, C);
  auto q = qkv_chunks[0];
  auto k = qkv_chunks[1];
  auto v = qkv_chunks[2];

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

  // 总的token长度，随着推理的进行，这个值每次加[1]
  auto total_seq_len = k.size(1);

  // 每个头att的形状都是 [T,total_seq_len],第一次推理应该是 [T,T]
  auto att = dense::Tensor::zeros(dense::kFloat32, {T, total_seq_len});

  // 输出张量，这是合并了多头之后的输出形状，等同于输入形状
  auto output = dense::Tensor::zeros_like(input);
  auto out_ptr = reinterpret_cast<float *>(output.data());

  float *q_ptr = reinterpret_cast<float *>(q.data());
  float *k_ptr = reinterpret_cast<float *>(k.data());
  float *v_ptr = reinterpret_cast<float *>(v.data());

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {
      header_forward(q_ptr, C, k_ptr, C, v_ptr, C, out_ptr, C, att, b, h);
    }
  }
  // output:[B,T,C]
  output = c_proj_->forward(output);
  output = resid_dropout_->forward(output);
  return output;
}

dense::Tensor CausalSelfAttention::backward(const dense::Tensor &grad_output) {
  // grad_output 形状: [B,T,C]
  auto grad_input = resid_dropout_->backward(grad_output);
  grad_input = c_proj_->backward(grad_input);

  auto B = grad_input.size(0);
  auto T = grad_input.size(1);
  auto C = grad_input.size(2);

  auto C3 = cached_qkv_.size(-1);

  auto grad_qkv = dense::Tensor::zeros_like(cached_qkv_);
  // 用于保存中间计算结果，最终会合并到 grad_qkv
  auto grad_att = dense::Tensor::zeros(dense::kFloat32, {T, T});

  for (size_t b = 0; b < B; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {

      header_backward(cached_qkv_, grad_qkv, grad_input, grad_att, b, h);
    }
  }

  return c_attn_->backward(grad_qkv);
}

void CausalSelfAttention::header_backward(dense::Tensor &qkv,
                                          dense::Tensor &grad_qkv,
                                          dense::Tensor &grad_input,
                                          dense::Tensor &grad_att, size_t b,
                                          size_t h) {
  auto B = grad_input.size(0);
  auto T = grad_input.size(1);
  auto C = grad_input.size(2);

  // 每次都要清零
  grad_att.zero_();
  auto grad_att_ptr = reinterpret_cast<float *>(grad_att.data());

  auto C3 = grad_qkv.size(-1);

  auto q_bt = reinterpret_cast<float *>(qkv.data()) + b * T * C3;
  auto k_bt = q_bt + C;
  auto v_bt = q_bt + C * 2;

  auto grad_q_bt = reinterpret_cast<float *>(grad_qkv.data()) + b * T * C3;
  auto grad_k_bt = grad_q_bt + C;
  auto grad_v_bt = grad_q_bt + C * 2;

  auto grad_input_bt = reinterpret_cast<float *>(grad_input.data()) + b * T * C;

  auto cached_att_after_dropout_bt =
      reinterpret_cast<float *>(cached_att_after_dropout_.data()) +
      b * config_.n_heads * T * T + h * T * T; // 每个批次每个头的起始位置

  auto cached_att_after_softmax_bt =
      reinterpret_cast<float *>(cached_att_after_softmax_.data()) +
      b * config_.n_heads * T * T + h * T * T; // 每个批次每个头的起始位置

  auto mask_ptr = reinterpret_cast<int8_t *>(mask_.data());

  // 计算 grad_att
  // grad_att = matmul(grad_input, v.transpose(-2, -1))
  dense::matmul_B_transpose(grad_input_bt + h * head_dim_, C,
                            v_bt + h * head_dim_, C3, nullptr, grad_att_ptr, T,
                            T /*M*/, T /*N*/, head_dim_ /*K*/);

  // 计算对 v 的梯度
  // grad_v = matmul(cached_att_after_dropout.transpose(-2, -1), grad_input)
  dense::matmul_A_transpose(
      cached_att_after_dropout_bt, T, grad_input_bt + h * head_dim_, C, nullptr,
      grad_v_bt + h * head_dim_, C3, T /*K*/, T /*M*/, head_dim_ /*N*/);

  auto grad_att_before_dropout = attn_dropout_->backward(grad_att);
  auto grad_att_before_dropout_ptr =
      reinterpret_cast<float *>(grad_att_before_dropout.data());

  // softmax 反向传播
  dense::mat_softmax_backward(grad_att_ptr, cached_att_after_softmax_bt,
                              grad_att_before_dropout_ptr, T, T);

  // mask fill, att[T,T]
  for (size_t i = 0; i < T; ++i) {
    for (size_t j = 0; j < T; ++j) {
      if (mask_ptr[i * config_.context_length + j] == 0) {
        grad_att_ptr[i * T + j] = 0.0f;
      }
    }
  }

  for (size_t i = 0; i < grad_att.numel(); ++i) {
    grad_att_ptr[i] *= scale_;
  }
  // 计算对 q 的梯度
  // grad_q = matmul(grad_att, k)
  dense::matmul(grad_att_ptr, T, k_bt + h * head_dim_, C3, nullptr,
                grad_q_bt + h * head_dim_, C3, T /*M*/, T /*K*/,
                head_dim_ /*N*/);

  // 计算对 k 的梯度
  // grad_k = matmul(grad_att.transpose(-2, -1), q)
  dense::matmul_A_transpose(grad_att_ptr, T, q_bt + h * head_dim_, C3, nullptr,
                            grad_k_bt + h * head_dim_, C3, T /*K*/, T /*M*/,
                            head_dim_ /*N*/);
}

Block::Block(Context *ctx, const std::string &instance_name,
             const ModelConfig &config, size_t idx)
    : Layer(ctx, instance_name), config_(config) {
  ln_1_ = std::make_unique<LayerNorm>(ctx, "ln_1", config_.emb_dim,
                                      config_.ln_epsilon, config_.qkv_bias);
  attn_ = std::make_unique<CausalSelfAttention>(ctx, "attn", config_, idx);
  ln_2_ = std::make_unique<LayerNorm>(ctx, "ln_2", config_.emb_dim,
                                      config_.ln_epsilon, config_.qkv_bias);
  mlp_ = std::make_unique<MLP>(ctx, "mlp", config_);
}

Block::~Block() = default;

dense::Tensor Block::forward(const dense::Tensor &input) {
  auto N = input.numel(); // 总元素数

  auto x = ln_1_->forward(input);
  x = attn_->forward(x);
  {
    // 第一个残差连接
    auto in_ptr = reinterpret_cast<const float *>(input.data());
    auto x_ptr = reinterpret_cast<float *>(x.data());
    for (size_t i = 0; i < N; ++i) {
      x_ptr[i] += in_ptr[i];
    }
  }
  auto after_first_residual = x;

  x = ln_2_->forward(x);
  x = mlp_->forward(x);
  {
    // 第二个残差连接
    auto in_ptr = reinterpret_cast<const float *>(after_first_residual.data());
    auto x_ptr = reinterpret_cast<float *>(x.data());
    for (size_t i = 0; i < N; ++i) {
      x_ptr[i] += in_ptr[i];
    }
  }
  return x;
}

dense::Tensor Block::backward(const dense::Tensor &grad_output) {
  auto N = grad_output.numel();

  auto grad_shortcut_2 = grad_output;

  auto grad_from_mlp = mlp_->backward(grad_output);
  auto grad_before_ln2 = ln_2_->backward(grad_from_mlp);
  {
    // 第二个残差连接的梯度
    auto shortcut_2_ptr = reinterpret_cast<float *>(grad_shortcut_2.data());
    auto grad_ptr = reinterpret_cast<float *>(grad_before_ln2.data());
    for (size_t i = 0; i < N; ++i) {
      grad_ptr[i] += shortcut_2_ptr[i];
    }
  }

  auto grad_shortcut_1 = grad_before_ln2;
  auto grad_from_attn = attn_->backward(grad_before_ln2);
  auto grad_before_ln1 = ln_1_->backward(grad_from_attn);
  {
    // 第一个残差连接的梯度
    auto shortcut_1_ptr = reinterpret_cast<float *>(grad_shortcut_1.data());
    auto grad_ptr = reinterpret_cast<float *>(grad_before_ln1.data());
    for (size_t i = 0; i < N; ++i) {
      grad_ptr[i] += shortcut_1_ptr[i];
    }
  }
  return grad_before_ln1;
}

LogSoftmaxCrossEntropyLoss::LogSoftmaxCrossEntropyLoss() = default;
LogSoftmaxCrossEntropyLoss::~LogSoftmaxCrossEntropyLoss() = default;

double LogSoftmaxCrossEntropyLoss::forward(const dense::Tensor &input,
                                           const dense::Tensor &target) {
  // input [B,T,vocab_size]
  // target 的形状是 [B,T] 或 [B,T,vocab_size]

  auto B = input.size(0); // 批大小
  auto T = input.size(1); // token数
  auto C = input.size(2); // 类别数

  if (target.dim() == 2) {
    if (target.dtype() != dense::DType::kInt32) {
      throw std::runtime_error("输入张量的类型必须是:kInt32");
    }

    // 如果 target 是整数标签，转换为 one-hot 编码
    // 创建一个全零的张量，作为 one-hot 编码的容器
    cached_target_ = dense::Tensor::zeros_like(input);

    for (size_t b = 0; b < B; ++b) {
      auto target_bt = reinterpret_cast<const int32_t *>(target.data()) + b * T;

      for (size_t t = 0; t < T; ++t) {
        auto y_true_bt = reinterpret_cast<float *>(cached_target_.data()) +
                         b * T * C + t * C;
        for (size_t k = 0; k < C; ++k) {
          if (target_bt[t] == k) {
            y_true_bt[k] = 1.0f; // one-hot 编码
            break;
          }
        }
      }
    }
  } else {
    cached_target_ = target; // 如果已经是 one-hot 编码，直接使用
  }

  cached_softmax_ = dense::Tensor::zeros_like(input);

  double total_loss = 0.0f;

  std::unique_ptr<float[]> z_shift(new float[C]);

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      auto in_bt =
          reinterpret_cast<const float *>(input.data()) + b * T * C + t * C;

      auto cached_softmax_bt =
          reinterpret_cast<float *>(cached_softmax_.data()) + b * T * C + t * C;

      auto y_true_bt = reinterpret_cast<const float *>(cached_target_.data()) +
                       b * T * C + t * C;

      // 计算 z_max
      float z_max = -INFINITY;
      for (size_t k = 0; k < C; ++k) {
        if (in_bt[k] > z_max) {
          z_max = in_bt[k];
        }
      }

      // exp_sum= sum(e^(z_k-z_max))
      float exp_sum = 0.0f;
      for (size_t k = 0; k < C; ++k) {
        // z_k-z_max
        z_shift[k] = in_bt[k] - z_max;
        // e^(z_k-z_max)
        exp_sum += std::exp(z_shift[k]);
      }

      // log_sum_exp = log(sum(e^(z_k-z_max)))
      float log_sum_exp = std::log(exp_sum);

      double sum_of_products = 0.0f;
      for (size_t k = 0; k < C; ++k) {
        // [z_k-z_max - log(sum(e^(z_k-z_max)))]
        auto pred_log_softmax = z_shift[k] - log_sum_exp;
        // cached_softmax_bt: pred_log_softmax 再取指数就还原为 softmax 值
        // 所以 cached_softmax_bt 存储的是 Softmax 计算值，用于反向传播
        cached_softmax_bt[k] = std::exp(pred_log_softmax);
        // sum(y_k* (pred_log_softmax))
        sum_of_products += y_true_bt[k] * pred_log_softmax;
      }
      total_loss += sum_of_products;
    }
  }
  // 损失函数 J = - sum(y_k* (pred_log_softmax))
  total_loss = -total_loss / static_cast<double>(B * T);
  return total_loss;
}

dense::Tensor LogSoftmaxCrossEntropyLoss::backward() {
  auto B = cached_softmax_.size(0); // 批大小
  auto T = cached_softmax_.size(1); // token数
  auto C = cached_softmax_.size(2); // 类别数

  auto grad = dense::Tensor::zeros_like(cached_softmax_);

  auto N = static_cast<double>(B * T);

  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      auto cached_softmax_bt =
          reinterpret_cast<const float *>(cached_softmax_.data()) + b * T * C +
          t * C;
      auto y_true_bt = reinterpret_cast<const float *>(cached_target_.data()) +
                       b * T * C + t * C;
      auto grad_bt = reinterpret_cast<float *>(grad.data()) + b * T * C + t * C;

      for (size_t k = 0; k < C; ++k) {
        // 计算梯度
        grad_bt[k] = (cached_softmax_bt[k] - y_true_bt[k]) / N;
      }
    }
  }
  // 形状 [B,T,C]
  return grad; // 返回平均化的梯度
}

AdamW::AdamW(double learning_rate, double beta1, double beta2, double epsilon,
             double weight_decay)
    : lr_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
      weight_decay_(weight_decay), step_(0) {}

void AdamW::ensure_state(int group_idx, int param_idx,
                         const dense::Tensor &param) {
  // 扩展容器大小
  if (m_states_.size() <= group_idx) {
    m_states_.resize(group_idx + 1);
    v_states_.resize(group_idx + 1);
  }
  if (m_states_[group_idx].size() <= param_idx) {
    m_states_[group_idx].resize(param_idx + 1);
    v_states_[group_idx].resize(param_idx + 1);
  }

  // 惰性初始化
  if (!m_states_[group_idx][param_idx].is_defined()) {
    m_states_[group_idx][param_idx] = dense::Tensor::zeros_like(param);
    v_states_[group_idx][param_idx] = dense::Tensor::zeros_like(param);
  }
}

void AdamW::update(ParamsAndGrads &params_and_grads) {
  ++step_;

  // 预计算偏差修正因子
  double bias_correction1 = 1.0 - std::pow(beta1_, step_);
  double bias_correction2 = 1.0 - std::pow(beta2_, step_);

  for (size_t i = 0; i < params_and_grads.weights.size(); ++i) {
    Group &param_group = params_and_grads.weights[i];
    GradGroup &grad_group = params_and_grads.grads[i];

    // 参数列表：[权重, 偏置]
    std::vector<std::pair<dense::Tensor, dense::Tensor>> param_grad_pairs = {
        {std::get<1>(param_group), std::get<1>(grad_group)}, // 权重
        {std::get<2>(param_group), std::get<2>(grad_group)}  // 偏置
    };

    for (size_t j = 0; j < param_grad_pairs.size(); ++j) {
      auto param = param_grad_pairs[j].first;
      auto grad = param_grad_pairs[j].second;

      if (param.is_defined() && grad.is_defined() && param.numel() > 0) {
        ensure_state(i, j, param);
        size_t N = param.numel();
        auto param_ptr = reinterpret_cast<float *>(param.data());
        auto grad_ptr = reinterpret_cast<float *>(grad.data());
        if (j == 0 && weight_decay_ != 0) {
          // 等价  p.mul_(1 - options.lr() * options.weight_decay());
          for (size_t k = 0; k < N; ++k) {
            param_ptr[k] -= lr_ * weight_decay_ * param_ptr[k];
            // param_ptr[i] *= (1 - lr_ * weight_decay_);
          }
          //*param *= (1 - lr_ * weight_decay_);
        }

        auto &m = m_states_[i][j];
        auto &v = v_states_[i][j];
        auto m_ptr = reinterpret_cast<float *>(m.data());
        auto v_ptr = reinterpret_cast<float *>(v.data());
        for (size_t k = 0; k < N; ++k) {
          m_ptr[k] = beta1_ * m_ptr[k] + (1 - beta1_) * grad_ptr[k];
          v_ptr[k] =
              beta2_ * v_ptr[k] + (1 - beta2_) * (grad_ptr[k] * grad_ptr[k]);
          auto m_hat = m_ptr[k] / bias_correction1;
          auto v_hat = v_ptr[k] / bias_correction2;
          auto adam_update = m_hat / (std::sqrt(v_hat) + epsilon_);
          param_ptr[k] -= lr_ * adam_update;
        }
        // 等价 param->addcdiv_(m_hat, v_hat.sqrt().add_(epsilon_), -lr_);
      }
    }
  }
}

GPT::GPT(const ModelConfig &config) : config_(config) {
  wte_ = std::make_unique<Embedding>(&ctx_, "wte", config_.vocab_size,
                                     config_.emb_dim, 50256);
  wpe_ = std::make_unique<Embedding>(&ctx_, "wpe", config_.context_length,
                                     config_.emb_dim);
  dropout_ =
      std::make_unique<Dropout>(&ctx_, "dropout", config_.drop_rate /*0.1*/);
  // 创建 n_layers 个 Block 实例
  for (size_t i = 0; i < config_.n_layers; ++i) {
    auto block = std::make_unique<Block>(&ctx_, "h", config_, i);
    h_.emplace_back(std::move(block));
  }
  ln_f_ = std::make_unique<LayerNorm>(&ctx_, "ln_f", config_.emb_dim,
                                      config_.ln_epsilon, true);
  lm_head_ = std::make_unique<Linear>(&ctx_, "lm_head", config_.emb_dim,
                                      config_.vocab_size, false);
}

GPT::~GPT() = default;

void GPT::init_weights() {
  wte_->W_.zero_();
  lm_head_->W_ = wte_->W_;

  for (auto &layer : ctx_.layers) {
    if (layer->instance_name() == "lm_head") {
      // lm_head 不需要初始化，因为它与嵌入层共享权重，我们在嵌入层初始化即可
      continue;
    }
    if (layer->W_.is_defined()) {
      layer->W_.zero_();
    }
    if (layer->b_.is_defined()) {
      layer->b_.zero_();
    }

    if (layer->name() == "Linear") {
      double stddev = 0.02;
      if (layer->instance_name() == "c_proj") {
        stddev = 0.02 / std::sqrtf(2.0 * config_.n_layers);
      }
      auto ptr = reinterpret_cast<float *>(layer->W_.data());
      normal_fill(ptr, layer->W_.numel(), 0.0, stddev);
      if (layer->b_.is_defined()) {
        auto b_ptr = reinterpret_cast<float *>(layer->b_.data());
        std::fill(b_ptr, b_ptr + layer->b_.numel(), 0.0f);
      }
    } else if (layer->name() == "Embedding") {
      auto ptr = reinterpret_cast<float *>(layer->W_.data());
      normal_fill(ptr, layer->W_.numel(), 0.0, 0.02);
    } else if (layer->name() == "LayerNorm") {
      if (layer->b_.is_defined()) {
        auto b_ptr = reinterpret_cast<float *>(layer->b_.data());
        std::fill(b_ptr, b_ptr + layer->b_.numel(), 0.0f);
      }
      auto w_ptr = reinterpret_cast<float *>(layer->W_.data());
      std::fill(w_ptr, w_ptr + layer->W_.numel(), 1.0f);
    }
  }
}

void GPT::from_pretrained(const std::string &filename) {
  if (!filename.empty()) {
    if (!dense::ModelParams::load(filename, &model_params_)) {
      throw std::runtime_error("加载预训练权重文件失败");
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
  if (!ctx_.cache_) {
    ctx_.cache_ = std::make_unique<DynamicCache>(config_.n_layers,
                                                 config_.context_length);
  }
}

void GPT::enable_training(bool enable) { ctx_.training = enable; }

bool GPT::is_enable_cache() const {
  return ctx_.cache_ != nullptr && !ctx_.training;
}

dense::Tensor GPT::forward(const dense::Tensor &input) {
  auto B = input.size(0);
  auto T = input.size(1);
  assert(T <= config_.context_length);

  std::vector<int32_t> pos_data(B * T);
  size_t current_pos = 0;
  if (is_enable_cache()) {
    current_pos = ctx_.cache_->get_seq_length();
  }
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      pos_data[b * T + t] = current_pos + static_cast<int32_t>(t);
    }
  }

  auto pos = dense::Tensor::from_blob(dense::kInt32, {B, T}, &pos_data[0]);

  auto tok_emb = wte_->forward(input);
  auto pos_emb = wpe_->forward(pos);
  auto tok_pos_emb = dense::Tensor::zeros_like(tok_emb);

  {
    auto N = tok_pos_emb.numel();
    auto tok_ptr = reinterpret_cast<float *>(tok_emb.data());
    auto pos_ptr = reinterpret_cast<float *>(pos_emb.data());
    auto out_ptr = reinterpret_cast<float *>(tok_pos_emb.data());
    for (size_t i = 0; i < N; ++i) {
      out_ptr[i] = tok_ptr[i] + pos_ptr[i];
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

dense::Tensor GPT::backward(const dense::Tensor &grad_output) {
  auto grad = lm_head_->backward(grad_output);
  grad = ln_f_->backward(grad);

  // 3. Transformer Block 层 `h_` 的反向传播（逆序遍历）
  // 从最后一个 Block 开始，到第一个 Block
  for (int64_t i = h_.size() - 1; i >= 0; --i) {
    auto &block = h_[i];
    // 每个 block 的 backward 接收来自上一个 block（或 ln_f_）的梯度，并返回对该
    // block 输入的梯度
    grad = block->backward(grad);
  }

  // grad 此时是损失对 (tok_emb + pos_emb) 之后，dropout 之前的输出的梯度
  grad = dropout_->backward(grad);

  // 5. 词嵌入和位置嵌入的反向传播
  // dropout_ 的输入是 tok_emb_cache_ + pos_emb_cache_
  // 因此，grad （即 dL/d_(tok_emb+pos_emb)）需要分别传递给 tok_emb 和 pos_emb
  // dL/d_tok_emb = dL/d_(tok_emb+pos_emb)
  // dL/d_pos_emb = dL/d_(tok_emb+pos_emb)

  // wte_ (Embedding) 的 backward
  // wte_ 的 backward 接收 dL/d_tok_emb
  auto grad_input = wte_->backward(grad); // 这是损失对原始 token input 的梯度

  // wpe_ (Embedding) 的 backward
  // wpe_ 的 backward 接收 dL/d_pos_emb
  // 注意：pos_emb 是通过位置索引生成的，通常位置嵌入不需要计算对原始 `pos`
  // 索引的梯度， 而是直接更新 `wpe_` 自身的权重。这里调用 `wpe_->backward`
  // 即可。
  wpe_->backward(grad); // 对位置嵌入权重的梯度会在这里计算

  return grad_input;
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
    // [B,T,C]
    auto B = logits.size(0);
    auto T = logits.size(1);
    auto C = logits.size(2);
    // 只取第一个批次的最后一个token预测向量
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

void GPT::clear_grads() {
  for (auto &layer : ctx_.layers) {
    if (layer->grad_W_.is_defined()) {
      layer->grad_W_.zero_();
    }
    if (layer->grad_b_.is_defined()) {
      layer->grad_b_.zero_();
    }
  }
}

void GPT::get_params_and_grads(ParamsAndGrads &params_and_grads) {
  for (auto &layer : ctx_.layers) {
    params_and_grads.weights.emplace_back(
        Group{layer->instance_name(), layer->W_, layer->b_});
    params_and_grads.grads.emplace_back(
        GradGroup{layer->instance_name(), layer->grad_W_, layer->grad_b_});
  }
}
