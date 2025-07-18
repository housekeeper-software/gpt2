#include <string>
#include "layers.h"
#include "safe_tensors.h"
#include "sampling.h"
#include "tensor.h"
#include "tiktokenizer.h"
#include <functional>
#include <iostream>

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

int main() {
  TikTokenizer tokenizer;
  tokenizer.load_vocabulary(model_dir + kVocabFile);
  tokenizer.load_merge_rules(model_dir + kMergesFile);
  tokenizer.add_special_tokens({"<|endoftext|>"});

  ModelConfig config;
  if (!config.InitFromFile(model_dir + kConfigFile)) {
    std::cerr << "Failed to initialize model config." << std::endl;
    return -1;
  }

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

  GPT gpt(config);

  gpt.from_pretrained(model_dir + kModelFile);
  //gpt.save(save_model_dir + "/gpt2-small.safetensors");

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
  return 0;
}