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
  float expansion_ratio;
  float ln_epsilon;
};

using Group = std::tuple<std::string, dense::Tensor, dense::Tensor>;
using GradGroup = std::tuple<std::string, dense::Tensor, dense::Tensor>;

struct ParamsAndGrads {
  std::vector<Group> weights;
  std::vector<GradGroup> grads;
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
            float epsilon = 1e-05, bool has_bias = false);
  ~LayerNorm() override;
  dense::Tensor forward(const dense::Tensor &input) override;
  dense::Tensor backward(const dense::Tensor &grad_output) override;
  std::string name() const override { return "LayerNorm"; }

private:
  float epsilon_;
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
  void header_forward(float *q, size_t q_stride, float *k, size_t k_stride,
                      float *v, size_t v_stride, float *out, size_t out_stride,
                      dense::Tensor &att, size_t b, size_t h);
  void header_backward(dense::Tensor &qkv, dense::Tensor &grad_qkv,
                       dense::Tensor &grad_input, dense::Tensor &grad_att,
                       size_t b, size_t h);

  size_t index_;
  float scale_;
  dense::Tensor cached_qkv_;
  dense::Tensor cached_att_after_softmax_, cached_att_after_dropout_;
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
  Block(const Block &) = delete;
  Block &operator=(const Block &) = delete;
};

class LogSoftmaxCrossEntropyLoss {
public:
  LogSoftmaxCrossEntropyLoss();
  ~LogSoftmaxCrossEntropyLoss();
  // target，如果是[B,T]，则类型为 int32_t，否则就是 float32 的
  // one-hot编码，形如[B,T,C]
  double forward(const dense::Tensor &input, const dense::Tensor &target);
  dense::Tensor backward();

private:
  dense::Tensor cached_target_;  // 缓存真实标签 (one-hot 编码)
  dense::Tensor cached_softmax_; // 缓存 Softmax 概率 (用于反向传播的简化)
};

class AdamW {
public:
  // 构造函数：初始化优化器参数
  AdamW(double learning_rate = 1e-4, double beta1 = 0.9, double beta2 = 0.999,
        double epsilon = 1e-8, double weight_decay = 0.01);

  // 更新模型参数的方法
  // params_and_grads 包含了所有可训练参数及其梯度
  void update(ParamsAndGrads &params_and_grads);

  void set_learning_rate(double lr) { lr_ = lr; }
  double get_learning_rate() const { // 可以添加一个获取当前学习率的方法
    return lr_;
  }

private:
  void ensure_state(int group_idx, int param_idx, const dense::Tensor &param);

  double lr_;
  double beta1_;
  double beta2_;
  double epsilon_;
  double weight_decay_;
  int step_; // 时间步，用于偏差修正

  // 存储每个参数的第一次矩估计 (m)
  std::vector<std::vector<dense::Tensor>> m_states_;
  // 存储每个参数的第二次矩估计 (v)
  std::vector<std::vector<dense::Tensor>> v_states_;
};

class CosineAnnealingWarmRestarts {
public:
  CosineAnnealingWarmRestarts(double initial_lr, int T_0, int T_mult = 1,
                              double eta_min = 0.0, int last_epoch = -1);
  void step(std::optional<double> epoch = std::nullopt);
  double get_lr() const;

private:
  int T_0_;    // 初始周期长度
  int T_mult_; // 周期长度乘数
  double
      T_cur_; // 当前周期的 epoch 计数，注意这里依然使用 int，因为它是步进计数
  double eta_min_;    // 最小学习率
  double initial_lr_; // 原始初始学习率
  double last_epoch_; // 总的 epoch 计数，改为 double 以支持浮点数
  int T_i_;           // 当前周期的总长度
};

class GPT {
public:
  GPT(const ModelConfig &config);
  ~GPT();
  void from_pretrained(const std::string &filename);

  // 训练模式，需要初始化权重
  void init_weights();

  void save(const std::string &filename);

  void enable_cache();

  bool is_enable_cache() const;

  void enable_training(bool enable);

  dense::Tensor GPT::forward(const dense::Tensor &input);

  std::vector<int> inference(std::vector<int> tokens, int max_length,
                             SamplingChain *chain,
                             std::function<bool(int)> token_callback = nullptr);

  void get_params_and_grads(ParamsAndGrads &params_and_grads);

  void clear_grads();

  dense::Tensor backward(const dense::Tensor &grad_output);

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
  Context ctx_;
};
#endif