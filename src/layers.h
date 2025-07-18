#ifndef LAYERS_H
#define LAYERS_H

#include "safe_tensors.h"
#include "tensor.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>

class DynamicCache;
class SamplingChain;

class ModelConfig {
public:
  ModelConfig();
  ~ModelConfig();
  ModelConfig(const ModelConfig &);
  ModelConfig &operator=(const ModelConfig &);
  bool InitFromFile(const std::string &config_file);
  int64_t vocab_size;
  int64_t context_length;
  int64_t emb_dim;
  int64_t n_heads;
  int64_t n_layers;
  float drop_rate;
  bool qkv_bias;
};

class Layer;

class Context {
public:
  Context();
  ~Context();
  void Register(Layer *layer);

  uint8_t *allocate(size_t size);

  std::vector<Layer *> layers;
  bool training = false;

  std::shared_ptr<dense::Storage> storage;

  std::unique_ptr<DynamicCache> cache_;

private:
  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;
};

class Layer {
public:
  Layer(Context *ctx, const std::string &instance_name)
      : ctx_(ctx), instance_name_(instance_name) {}
  virtual ~Layer() = default;
  virtual std::string name() const { return "Layer"; }
  std::string instance_name() const { return instance_name_; }

  size_t data_size() const { return W_.data_size() + b_.data_size(); }

  bool is_training() const { return ctx_->training; }

  virtual dense::Tensor forward(const dense::Tensor &input) = 0;
  virtual dense::Tensor backward(const dense::Tensor &grad_output) {
    return dense::Tensor();
  }

  dense::Tensor W_;
  dense::Tensor b_;
  dense::Tensor grad_W_;
  dense::Tensor grad_b_;

  Context *ctx_;

private:
  std::string instance_name_;
  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
};

class Embedding : public Layer {
public:
  Embedding(Context *ctx, const std::string &instance_name,
            int64_t num_embeddings, int64_t embedding_dim,
            int64_t padding_idx = -1);
  ~Embedding() override;

  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "Embedding"; }

private:
  void LazyInit();
  dense::Tensor input_cache_;
  int64_t num_embeddings_;
  int64_t embedding_dim_;
  int64_t padding_idx_;

  Embedding(const Embedding &) = delete;
  Embedding &operator=(const Embedding &) = delete;
};

class Linear : public Layer {
public:
  Linear(Context *ctx, const std::string &instance_name, int64_t in_features,
         int64_t out_features, bool has_bias = true);
  ~Linear() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "Linear"; }

private:
  void LazyInit();
  int64_t in_features_;
  int64_t out_features_;
  bool has_bias_;
  dense::Tensor input_cache_;
  Linear(const Linear &) = delete;
  Linear &operator=(const Linear &) = delete;
};

class LayerNorm : public Layer {
public:
  LayerNorm(Context *ctx, const std::string &instance_name, int64_t ndim,
            bool has_bias = false);
  ~LayerNorm() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "LayerNorm"; }

private:
  void LazyInit();
  int64_t ndim_; // 存储维度大小
  bool has_bias_;
  dense::Tensor input_cache_;
  dense::Tensor mean_;
  dense::Tensor rstd_;
  LayerNorm(const LayerNorm &) = delete;
  LayerNorm &operator=(const LayerNorm &) = delete;
};

class GELU : public Layer {
public:
  GELU(Context *ctx, const std::string &instance_name);
  ~GELU() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "GELU"; }

private:
  dense::Tensor input_cache_;
  GELU(const GELU &) = delete;
  GELU &operator=(const GELU &) = delete;
};

class Dropout : public Layer {
public:
  Dropout(Context *ctx, const std::string &instance_name,
          float dropout_ratio = 0.5);
  ~Dropout() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "Dropout"; }

private:
  float dropout_ratio_; // 设置为零的神经元比例（丢弃率）
  float scale_;         // 缩放因子，用于 inverted dropout
  dense::Tensor mask_;  // 缓存丢弃掩码，用于反向传播

  Dropout(const Dropout &) = delete;
  Dropout &operator=(const Dropout &) = delete;
};

// 多层感知机（MLP）层
// 该层通常用于 Transformer 模型中的前馈网络部分
class MLP : public Layer {
public:
  MLP(Context *ctx, const std::string &instance_name,
      const ModelConfig &config);
  ~MLP() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "MLP"; }
  // MLP 的配置参数
  // 包括嵌入维度、dropout 比例等
  ModelConfig config_;
  std::unique_ptr<Linear> c_fc_;
  std::unique_ptr<Linear> c_proj_;
  std::unique_ptr<Dropout> dropout_;
  std::unique_ptr<GELU> gelu_;

private:
  MLP(const MLP &) = delete;
  MLP &operator=(const MLP &) = delete;
};

// Causal Self-Attention 层
// 该层实现了自回归的注意力机制，通常用于语言模型
class CausalSelfAttention : public Layer {
public:
  CausalSelfAttention(Context *ctx, const std::string &instance_name,
                      const ModelConfig &config, size_t idx);
  ~CausalSelfAttention() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "CausalSelfAttention"; }

  ModelConfig config_;
  int64_t head_dim_;
  std::unique_ptr<Linear> c_attn_;
  std::unique_ptr<Linear> c_proj_;
  std::unique_ptr<Dropout> attn_dropout_;
  std::unique_ptr<Dropout> resid_dropout_;
  dense::Tensor mask_;

private:
  dense::Tensor forward_cache(const dense::Tensor &input);
  size_t index_;
  dense::Tensor cached_q_, cached_k_, cached_v_;
  dense::Tensor cached_att_before_softmax_, cached_att_after_dropout_;
  CausalSelfAttention(const CausalSelfAttention &) = delete;
  CausalSelfAttention &operator=(const CausalSelfAttention &) = delete;
};

class Block : public Layer {
public:
  Block(Context *ctx, const std::string &instance_name,
        const ModelConfig &config, size_t idx);
  ~Block() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "Block"; }

  ModelConfig config_;
  std::unique_ptr<LayerNorm> ln_1_;
  std::unique_ptr<CausalSelfAttention> attn_;
  std::unique_ptr<LayerNorm> ln_2_;
  std::unique_ptr<MLP> mlp_;

private:
  dense::Tensor
      shortcut_1_cache_; // 缓存第一个残差连接的 shortcut (即原始 input)
  dense::Tensor shortcut_2_cache_; // 缓存第二个残差连接的 shortcut (即 attn_
                                   // output + shortcut_1_cache_)
  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;
};

class GPT {
public:
  GPT(const ModelConfig &config);
  ~GPT();
  void from_pretrained(const std::string &filename);

  void save(const std::string &filename);

  void enable_cache();

  bool is_enable_cache() const;

  void enable_training(bool enable);

  dense::Tensor GPT::forward(const dense::Tensor &input);

  std::vector<int> inference(std::vector<int> tokens, int max_length,
                             SamplingChain *chain,
                             std::function<bool(int)> token_callback = nullptr);

  dense::Tensor shared_weight_;
  // token 嵌入
  std::unique_ptr<Embedding> wte_;
  // 位置嵌入
  std::unique_ptr<Embedding> wpe_;
  // dropout
  std::unique_ptr<Dropout> dropout_;
  // Transformer 块列表
  std::vector<std::unique_ptr<Block>> h_;
  // 最终的 LayerNorm
  std::unique_ptr<LayerNorm> ln_f_;
  // 语言模型头
  std::unique_ptr<Linear> lm_head_;

private:
  void _load_weights();

  static size_t _write_tensor(dense::ModelParams &model_params,
                              const std::string &name,
                              const dense::Tensor &tensor);
  ModelConfig config_;
  dense::ModelParams model_params_;
  std::unique_ptr<Context> ctx_;
};
#endif