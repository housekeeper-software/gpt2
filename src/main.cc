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
  std::vector<int64_t> y = {1, 3};

  dense::Tensor x_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {1, 2, 4}, &x[0]);
  dense::Tensor y_tensor =
      dense::Tensor::from_blob(dense::DType::kFloat32, {1, 2}, &y[0]);

  LogSoftmaxCrossEntropyLoss criterion;
  double loss = criterion.forward(x_tensor, y_tensor);
  std::cout << "loss:" << loss << std::endl;
  auto output = criterion.backward();
  std::cout << output.to_string() << std::endl;
}

// #define INFERENCE

int main() {
#if 0
  // test_layer_normal();
  // test_gelu();
  // test_mat_softmax();
  test_logsoftmax_crossentroy();
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
  int token_size = 80;
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
  train(&gpt, &train_data_loader, &test_data_loader, &loss, &optimizer, 10,
        save_model_dir, 4);
#endif
  return 0;
}