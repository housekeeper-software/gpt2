#include "data_loader.h"
#include "layers.h"
#include "safe_tensors.h"
#include "sampling.h"
#include "tensor.h"
#include "tiktokenizer.h"
#include "training.h"
#include <functional>
#include <iostream>
#include <string>

const std::string kDataDir = "C:/dev/llm/c/gpt/gpt-2/data";

const std::string model_dir = "C:/dev/llm/c/gpt/gpt-2/data/gpt2-small/";
const std::string save_model_dir = "C:/dev/llm/c/gpt/gpt-c/model";
const char kVocabFile[] = "vocab.json";
const char kMergesFile[] = "merges.txt";
const char kModelFile[] = "model.safetensors";
const char kConfigFile[] = "config.json";

bool simple_display(int token_id, std::string &decoded_cache,
                    TikTokenizer *tokenizer) {
  if (token_id == 50256)
    return false;
  std::string decoded = tokenizer->decode({token_id});
  decoded_cache.append(decoded);
  if (decoded.back() == '\n') {
    if (decoded_cache.length() > 4 &&
        decoded_cache.substr(decoded_cache.length() - 2) == "\n\n") {
      // 最后两个字符是换行符
      return false;
    }
  }
  std::cout << decoded;
  std::cout.flush(); // 立即显示
  return true;
}

void test_layer_normal() {
  Context ctx;
  ctx.training = true;
  LayerNorm l(&ctx, "norm", 4, true);

  std::vector<float> x = {1.0,  2.0,  3.0,  4.0,  5.0,  6.1,  7.5,  8.0,
                          9.0,  10.4, 11.3, 12.0, 13.0, 14.0, 15.0, 16.0,
                          17.0, 18.5, 19.1, 20.0, 21.0, 22.3, 23.6, 24.0};
  std::vector<float> g = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};

  std::vector<float> w = {1.1, 1.2, 1.3, 1.4};
  std::vector<float> b = {0.5, 0.6, 0.7, 0.8};
  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 3, 4}, &x[0]);
  dense::Tensor g_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 3, 4}, &g[0]);
  dense::Tensor w_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {4}, &w[0]);
  dense::Tensor b_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {4}, &b[0]);
  l.W_ = w_tensor;
  l.b_ = b_tensor;
  auto result = l.forward(x_tensor);
  std::cout << result.to_string() << std::endl;
  auto output = l.backward(g_tensor);
  std::cout << output.to_string() << std::endl;
  int m = 0;
}

void test_gelu() {
  Context ctx;
  ctx.training = true;
  GELU gelu(&ctx, "gelu");
  std::vector<float> x = {1.0,  2.0,  3.0,  4.0,  5.0,  6.1,  7.5,  8.0,
                          9.0,  10.4, 11.3, 12.0, 13.0, 14.0, 15.0, 16.0,
                          17.0, 18.5, 19.1, 20.0, 21.0, 22.3, 23.6, 24.0};
  std::vector<float> g = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                          0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
                          1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4};
  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 3, 4}, &x[0]);
  dense::Tensor g_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 3, 4}, &g[0]);
  auto result = gelu.forward(x_tensor);
  std::cout << result.to_string() << std::endl;
  auto output = gelu.backward(g_tensor);
  std::cout << output.to_string() << std::endl;
}

void test_mat_softmax() {
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.1, 7.5, 8.0};
  std::vector<float> g = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};

  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 4}, &x[0]);
  dense::Tensor g_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {2, 4}, &g[0]);

  auto x_ptr = reinterpret_cast<float *>(x_tensor.data());
  auto g_ptr = reinterpret_cast<float *>(g_tensor.data());

  dense::mat_softmax_forward(x_ptr, 2, 4);
  std::cout << x_tensor.to_string() << std::endl;

  dense::Tensor out = dense::Tensor::zeros(dense::DType::kFloat32, {2, 4});
  auto g_out_ptr = reinterpret_cast<float *>(out.data());
  dense::mat_softmax_backward(g_out_ptr, x_ptr, g_ptr, 2, 4);
  std::cout << out.to_string() << std::endl;
}

void test_logsoftmax_crossentroy() {
  std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 5.0, 6.1, 7.5, 8.0};
  std::vector<int32_t> y = {1, 3};

  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {1, 2, 4}, &x[0]);
  dense::Tensor y_tensor =
      dense::Tensor::from_blob(dense::DType::kInt32, {1, 2}, &y[0]);

  LogSoftmaxCrossEntropyLoss criterion;
  double loss = criterion.forward(x_tensor, y_tensor);
  std::cout << "loss:" << loss << std::endl;
  auto output = criterion.backward();
  std::cout << output.to_string() << std::endl;
}

void test_self_attension() {
  ModelConfig config;
  config.context_length = 4;
  config.n_heads = 2;
  config.emb_dim = 4;
  Context ctx;
  ctx.training = true;

  Linear lm(&ctx, "linear", config.emb_dim, config.emb_dim);
  CausalSelfAttention att(&ctx, "att", config, 0);
  LogSoftmaxCrossEntropyLoss criterion;

  std::vector<float> x = {1.0, 2.0, 3.0, 4.0, 2.1, 3.1, 4.1, 5.1};
  std::vector<int32_t> y = {1, 3};

  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {1, 2, 4}, &x[0]);
  dense::Tensor y_tensor =
      dense::Tensor::from_blob(dense::DType::kInt32, {1, 2}, &y[0]);

  std::vector<float> lm_weight = {
      -0.5306, -0.3244, 1.7593,  -0.4040, 2.1890, -0.8990, 0.5889, -0.3356,
      -0.1402, 1.0750,  -0.0240, -0.5482, 0.9018, -0.1813, 0.3789, -0.0610};

  std::vector<float> lm_bias = {1.6170, -1.3437, -0.4405, 0.9480};
  std::vector<float> c_attn_weight = {
      -0.3018, 1.4686,  0.4823,  0.5763,  -1.0916, 0.9300,  -0.9224, -0.0236,
      -0.6322, -0.1761, 0.7195,  1.2235,  -0.4291, -0.6773, 0.0332,  -1.3214,
      0.6860,  -0.7695, 0.6082,  1.4022,  -1.1864, -1.0105, -1.1372, -2.3769,
      -0.3113, -0.5944, -2.6337, -0.7431, 0.6553,  0.3310,  -1.3414, 0.0925,
      -1.1995, 0.1116,  -0.0292, 0.8146,  0.0358,  -2.1648, 0.2887,  -1.0243,
      -1.6303, -0.4749, 1.8948,  -0.5505, 0.3283,  0.2875,  -0.2010, 0.2324};
  std::vector<float> c_attn_bias = {0.9123, 1.8398,  0.2893, -2.8144,
                                    0.0060, 1.7524,  1.7609, -0.9690,
                                    0.7582, -0.2516, 1.1334, 0.7132};
  std::vector<float> c_proj_weight = {
      -1.7952, 0.1719,  -2.0076, -0.6997, -0.3000, 0.2866,  -0.4165, 0.2632,
      -0.1933, -0.5491, -0.5827, -1.3250, -1.1561, -1.7588, 0.1984,  -0.1549};
  std::vector<float> c_proj_bias = {-1.0506, 0.7706, -1.3649, 0.6076};

  lm.W_ =
      dense::Tensor::from_blob(dense::DType::kFloat32, {4, 4}, &lm_weight[0]);
  lm.b_ = dense::Tensor::from_blob(dense::DType::kFloat32, {4}, &lm_bias[0]);
  att.c_attn_->W_ = dense::Tensor::from_blob(dense::DType::kFloat32, {12, 4},
                                             &c_attn_weight[0]);
  att.c_attn_->b_ =
      dense::Tensor::from_blob(dense::DType::kFloat32, {12}, &c_attn_bias[0]);
  att.c_proj_->W_ =
      dense::Tensor::from_blob(dense::kFloat32, {4, 4}, &c_proj_weight[0]);
  att.c_proj_->b_ =
      dense::Tensor::from_blob(dense::DType::kFloat32, {4}, &c_proj_bias[0]);

  auto out = lm.forward(x_tensor);
  std::cout << "lm forward output: " << out.to_string() << std::endl;
  auto logits = att.forward(out);
  std::cout << "att forward output: " << logits.to_string() << std::endl;

  auto loss = criterion.forward(logits, y_tensor);
  std::cout << "criterion loss: " << loss << std::endl;

  auto grad = criterion.backward();
  std::cout << "criterion backward output: " << grad.to_string() << std::endl;

  auto att_grad = att.backward(grad);
  std::cout << "att.c_attn_w_grad backward output: "
            << att.c_attn_->grad_W_.to_string() << std::endl;
  std::cout << "att.c_attn_b_grad backward output: "
            << att.c_attn_->grad_b_.to_string() << std::endl;

  std::cout << "att.c_proj_w_grad backward output: "
            << att.c_proj_->grad_W_.to_string() << std::endl;
  std::cout << "att.c_proj_b_grad backward output: "
            << att.c_proj_->grad_b_.to_string() << std::endl;

  std::cout << "att backward output: " << att_grad.to_string() << std::endl;
  auto lm_grad = lm.backward(att_grad);

  std::cout << "lm.grad_w backward output: " << lm.grad_W_.to_string()
            << std::endl;
  std::cout << "lm.grad_g backward output: " << lm.grad_b_.to_string()
            << std::endl;

  std::cout << "lm backward output: " << lm_grad.to_string() << std::endl;
  printf("done");
}

void test_all_matmul_functions(size_t M, size_t K, size_t N) {
  const float EPSILON = 1e-5;
  bool success = true;

  // 内存分配
  std::vector<float> A_vec(M * K);
  std::vector<float> B_vec(K * N);
  std::vector<float> bias_vec(N);
  std::vector<float> C_openblas_vec(M * N);
  std::vector<float> C_native_vec(M * N);

  // 数据初始化
  std::mt19937 gen(42); // 使用固定种子以确保可重现性
  std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
  auto fill_random = [&](std::vector<float> &vec) {
    for (auto &val : vec) {
      val = dis(gen);
    }
  };
  fill_random(A_vec);
  fill_random(B_vec);
  fill_random(bias_vec);

  // ================== 测试 1: C += A * B ==================
  std::cout << "Testing C += A * B..." << std::endl;
  std::fill(C_openblas_vec.begin(), C_openblas_vec.end(), 0.0f);
  std::fill(C_native_vec.begin(), C_native_vec.end(), 0.0f);

  dense::matmul_openblas(A_vec.data(), K, B_vec.data(), N, bias_vec.data(),
                         C_openblas_vec.data(), N, M, K, N);
  dense::matmul_native(A_vec.data(), K, B_vec.data(), N, bias_vec.data(),
                       C_native_vec.data(), N, M, K, N);

  for (size_t i = 0; i < M * N; ++i) {
    if (std::abs(C_openblas_vec[i] - C_native_vec[i]) > EPSILON) {
      std::cerr << "Test failed for C = A * B at index " << i << std::endl;
      std::cerr << "OpenBLAS: " << C_openblas_vec[i]
                << ", Native: " << C_native_vec[i] << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Test passed." << std::endl;
  }

  // ================== 测试 2: C += A_trans * B ==================
  std::cout << "Testing C += A_trans * B..." << std::endl;
  std::fill(C_openblas_vec.begin(), C_openblas_vec.end(), 0.0f);
  std::fill(C_native_vec.begin(), C_native_vec.end(), 0.0f);

  // 对于 A^T * B，A的形状变为[K, M]，B的形状为[K, N]
  std::vector<float> A_trans_vec(K * M);
  for (size_t k = 0; k < K; ++k) {
    for (size_t m = 0; m < M; ++m) {
      A_trans_vec[k * M + m] = A_vec[m * K + k];
    }
  }

  dense::matmul_A_transpose_openblas(A_trans_vec.data(), M, B_vec.data(), N,
                                     bias_vec.data(), C_openblas_vec.data(), N,
                                     K, M, N);
  dense::matmul_A_transpose_native(A_trans_vec.data(), M, B_vec.data(), N,
                                   bias_vec.data(), C_native_vec.data(), N, K,
                                   M, N);

  for (size_t i = 0; i < M * N; ++i) {
    if (std::abs(C_openblas_vec[i] - C_native_vec[i]) > EPSILON) {
      std::cerr << "Test failed for C = A_trans * B at index " << i
                << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Test passed." << std::endl;
  }

  // ================== 测试 3: C += A * B_trans ==================
  std::cout << "Testing C += A * B_trans..." << std::endl;
  std::fill(C_openblas_vec.begin(), C_openblas_vec.end(), 0.0f);
  std::fill(C_native_vec.begin(), C_native_vec.end(), 0.0f);

  // 对于 A * B^T，A的形状为[M, K]，B的形状变为[N, K]
  std::vector<float> B_trans_vec(N * K);
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      B_trans_vec[n * K + k] = B_vec[k * N + n];
    }
  }

  dense::matmul_B_transpose_openblas(A_vec.data(), K, B_trans_vec.data(), K,
                                     bias_vec.data(), C_openblas_vec.data(), N,
                                     M, N, K);
  dense::matmul_B_transpose_native(A_vec.data(), K, B_trans_vec.data(), K,
                                   bias_vec.data(), C_native_vec.data(), N, M,
                                   N, K);

  for (size_t i = 0; i < M * N; ++i) {
    if (std::abs(C_openblas_vec[i] - C_native_vec[i]) > EPSILON) {
      std::cerr << "Test failed for C = A * B_trans at index " << i
                << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Test passed." << std::endl;
  }

  // ================== 测试 4: C += A_trans * B_trans ==================
  std::cout << "Testing C += A_trans * B_trans..." << std::endl;
  std::fill(C_openblas_vec.begin(), C_openblas_vec.end(), 0.0f);
  std::fill(C_native_vec.begin(), C_native_vec.end(), 0.0f);

  // 对于 A^T * B^T，A的形状为[K, M]，B的形状为[N, K]
  dense::matmul_A_B_transpose_openblas(A_trans_vec.data(), M,
                                       B_trans_vec.data(), K, bias_vec.data(),
                                       C_openblas_vec.data(), N, K, M, N);
  dense::matmul_A_B_transpose_native(A_trans_vec.data(), M, B_trans_vec.data(),
                                     K, bias_vec.data(), C_native_vec.data(), N,
                                     K, M, N);

  for (size_t i = 0; i < M * N; ++i) {
    if (std::abs(C_openblas_vec[i] - C_native_vec[i]) > EPSILON) {
      std::cerr << "Test failed for C = A_trans * B_trans at index " << i
                << std::endl;
      success = false;
      break;
    }
  }
  if (success) {
    std::cout << "Test passed." << std::endl;
  }

  if (success) {
    std::cout << "All tests passed for dimensions M=" << M << ", K=" << K
              << ", N=" << N << std::endl;
  } else {
    std::cerr << "Some tests failed for dimensions M=" << M << ", K=" << K
              << ", N=" << N << std::endl;
  }
}

// #define INFERENCE

int main() {
#if 0
  // test_layer_normal();
  // test_gelu();
  // test_mat_softmax();
  // test_logsoftmax_crossentroy();
  //test_self_attension();
  test_all_matmul_functions(64,128,256);
  return 0;
#endif

  TikTokenizer tokenizer;
  tokenizer.load_vocabulary(model_dir + kVocabFile);
  tokenizer.load_merge_rules(model_dir + kMergesFile);
  tokenizer.add_special_tokens({"<|endoftext|>"});

  ModelConfig config;
  if (!config.InitFromFile(model_dir + kConfigFile)) {
    std::cerr << "Failed to initialize model config." << std::endl;
    return -1;
  }

  GPT gpt(config);

#if defined(INFERENCE)

  std::vector<std::string> breakers = {
      "\n",            // 换行符
      "\n\n",          // 双换行符，通常表示新段落
      ":",             // 冒号，常用于分隔键值对或对话
      "\"",            // 双引号，用于字符串或引用
      "'",             // 单引号
      " ",             // 空格
      ". ",            // 句号后跟空格
      "! ",            // 感叹号后跟空格
      "? ",            // 问号后跟空格
      "<|endoftext|>", // GPT 风格的结束 Token
      "<|im_start|>",  // GPT 风格的起始 Token
      "<|im_end|>",    // GPT 风格的结束 Token
      "###",           // Markdown 标题
      "```",           // Markdown 代码块
      "/*",            // C/C++ 多行注释开始
      "*/"             // C/C++ 多行注释结束
  };
  auto processed_breakers = tokenizer.process_breakers(breakers);

  gpt.from_pretrained(model_dir + kModelFile);
  // gpt.save(save_model_dir + "/gpt2-small.safetensors");

  SamplingChain smpl;
  smpl.add(std::make_unique<PenaltiesSampling>(512, 1.1f, 0.05f, 0.0f));
  smpl.add(std::make_unique<DrySampling>(config.context_length,
                                         0.8f,  // dry_multiplier
                                         1.75f, // dry_base
                                         2,     // dry_allowed_length
                                         128,   // dry_penalty_last_n
                                         processed_breakers));
  smpl.add(std::make_unique<TemperatureSampling>(0.1f));
  smpl.add(std::make_unique<TopPSampling>(0.1f, 1));
  smpl.add(std::make_unique<DistSampling>(LLAMA_DEFAULT_SEED));

  std::string test_str = "What is the capital city of France?";
  std::vector<int> encoded_ids = tokenizer.encode(test_str);
  gpt.enable_cache();
  gpt.enable_training(false);
  std::string decoded_cache;

  auto result = gpt.inference(encoded_ids, 300, &smpl,
                              std::bind(simple_display, std::placeholders::_1,
                                        decoded_cache, &tokenizer));
  std::string decoded_str = tokenizer.decode(result);
#else
  int token_size = 256;
  auto train_tensor =
      LoadPreProcessedData(kDataDir + "/wikitext-2-raw/wiki.train.bin");
  auto test_tensor =
      LoadPreProcessedData(kDataDir + "/wikitext-2-raw/wiki.test.bin");
  auto valid_tensor =
      LoadPreProcessedData(kDataDir + "/wikitext-2-raw/wiki.valid.bin");
  auto train_dataset =
      std::make_unique<TextDataset>(std::move(train_tensor), token_size);
  auto test_dataset =
      std::make_unique<TextDataset>(std::move(test_tensor), token_size);
  auto valid_dataset =
      std::make_unique<TextDataset>(std::move(valid_tensor), token_size);

  DataLoader train_data_loader(std::move(train_dataset), 4, true);
  DataLoader test_data_loader(std::move(test_dataset), 4, false);
  DataLoader valid_data_loader(std::move(valid_dataset), 4, false);
  LogSoftmaxCrossEntropyLoss loss;
  AdamW optimizer;
  gpt.init_weights();

  CosineAnnealingWarmRestarts scheduler(optimizer.get_learning_rate(), 10, 1,
                                        1e-6);
  TrainingArguments args;
  args.epochs = 100;
  args.accumulation_steps = 16;
  args.max_grad_norm = 1.0f;
  args.patience = 10;
  train(&gpt, save_model_dir, &train_data_loader, &test_data_loader, &loss,
        &optimizer, &scheduler, args);
#endif
  return 0;
}