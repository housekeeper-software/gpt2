#include "data_loader.h"
#include "tiktokenizer.h"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
std::string strip(const std::string &str) {
  const std::string whitespace = " \t\n\r\f\v";
  size_t start = str.find_first_not_of(whitespace);
  if (start == std::string::npos)
    return "";
  size_t end = str.find_last_not_of(whitespace);
  return str.substr(start, end - start + 1);
}

std::vector<int64_t> load_and_tokenize(TikTokenizer *tokenizer,
                                       const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "错误: 无法打开数据集: " << filename << std::endl;
    return {};
  }
  std::string line;
  std::vector<int64_t> all_tokens;
  all_tokens.reserve(1024 * 1024 * 10);
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    auto stripped_line = strip(line);
    if (stripped_line.empty() || stripped_line == "= = = <unk> = = =" ||
        stripped_line == "= = = = =")
      continue;
    auto tokens = tokenizer->encode(stripped_line);
    if (tokens.empty())
      continue;
    all_tokens.insert(all_tokens.end(), tokens.begin(), tokens.end());
    all_tokens.push_back(50256);
  }
  return all_tokens;
}

bool write_vector_to_binary_file(const std::string &filename,
                                 const std::vector<int64_t> &data) {
  // std::ios::binary 标志表示以二进制模式打开文件
  std::ofstream file(filename, std::ios::out | std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "错误: 无法打开文件 " << filename << " 进行写入。"
              << std::endl;
    return false;
  }

  // 写入vector的大小（可选，但推荐，方便读取时知道要读多少数据）
  size_t data_size = data.size();
  file.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

  // 写入vector的实际数据
  // file.write(const char* s, streamsize n);
  // s: 指向要写入数据的指针
  // n: 要写入的字节数
  file.write(reinterpret_cast<const char *>(data.data()),
             data_size * sizeof(int64_t));

  file.close();
  std::cout << "数据成功写入到二进制文件: " << filename << std::endl;
  return true;
}

// 读取函数
bool load_vector_from_binary_file(const std::string &filename,
                                  std::vector<int64_t> &data) {
  // std::ios::in 标志表示以输入模式打开文件
  // std::ios::binary 标志表示以二进制模式打开文件
  std::ifstream file(filename, std::ios::in | std::ios::binary);

  if (!file.is_open()) {
    std::cerr << "错误: 无法打开文件 " << filename << " 进行读取。"
              << std::endl;
    return false;
  }

  size_t data_size = 0;
  // 读取写入时的大小信息
  file.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

  if (file.fail()) {
    std::cerr << "错误: 读取文件大小失败或文件已损坏。" << std::endl;
    file.close();
    return false;
  }

  // 根据读取到的大小信息调整vector的大小
  data.resize(data_size);

  // 读取vector的实际数据
  file.read(reinterpret_cast<char *>(data.data()), data_size * sizeof(int64_t));

  if (file.fail()) {
    std::cerr << "错误: 读取数据失败或文件已损坏。" << std::endl;
    file.close();
    return false;
  }

  file.close();
  std::cout << "数据成功从二进制文件: " << filename << " 加载。" << std::endl;
  return true;
}
} // namespace

bool PreProcessData(const std::string &in_file, const std::string &out_file,
                    TikTokenizer *tokenizer) {
  auto result = load_and_tokenize(tokenizer, in_file);
  if (result.empty())
    return false;
  return write_vector_to_binary_file(out_file, result);
}

dense::Tensor LoadPreProcessedData(const std::string &filename) {
  std::vector<int64_t> all_tokens;
  load_vector_from_binary_file(filename, all_tokens);
  if (all_tokens.empty())
    return dense::Tensor();
  int64_t size = all_tokens.size();
  std::vector<int32_t> output;
  output.insert(output.end(), all_tokens.begin(), all_tokens.end());
  return dense::Tensor::from_blob(dense::DType::kInt32, {size}, &output[0])
      .clone();
}

TextDataset::TextDataset(dense::Tensor data_sequence, int context_length)
    : data_sequence_(std::move(data_sequence)),
      context_length_(context_length) {
  if (data_sequence_.size(0) < context_length_ + 1) {
    throw std::runtime_error(
        "data_sequence too short for the given context_length.");
  }
}

size_t TextDataset::size() const {
  // 减去 context_length，因为每个样本需要 context_length + 1 个 token
  // 例如，10个token，context_length=3，能形成 10 - 3 = 7 个样本
  // [0,1,2,3,4,5,6,7,8,9]
  // [0,1,2,3] -> idx=0
  // [1,2,3,4] -> idx=1
  // ...
  // [6,7,8,9] -> idx=6
  // size(0) - context_length
  return data_sequence_.size(0) - context_length_;
}

std::pair<dense::Tensor, dense::Tensor> TextDataset::get(size_t idx) const {
  size_t start_idx = idx;
  size_t end_idx = idx + context_length_ + 1;
  if (end_idx > data_sequence_.size(0)) {
    // 这意味着我们到达了数据集的末尾，无法形成完整的full_slice
    // 实际应用中，你可能需要决定如何处理：
    // 1. 抛出错误 (当前行为)
    // 2. 截断 full_slice (可能导致 batch 大小不一致)
    // 3. 循环回数据开头 (对于无限数据集)
    throw std::out_of_range("Dataset index out of bounds for full_slice.");
  }

  // full_slice = self.data_sequence[idx : idx + self.context_length + 1]
  size_t element_size = dense::get_element_size(data_sequence_.dtype());

  auto data_ptr = data_sequence_.data() + start_idx * element_size;

  auto x_input = dense::Tensor::from_blob(
      dense::DType::kInt32, {context_length_}, const_cast<uint8_t *>(data_ptr));
  data_ptr += element_size;
  auto y_target = dense::Tensor::from_blob(
      dense::DType::kInt32, {context_length_}, const_cast<uint8_t *>(data_ptr));
  return {x_input, y_target};
}

DataLoader::DataLoader(std::unique_ptr<TextDataset> dataset, size_t batch_size,
                       bool shuffle)
    : dataset_(std::move(dataset)), batch_size_(batch_size), shuffle_(shuffle),
      rng_(std::random_device{}()) {
  if (dataset_->size() < 1) {
    throw std::runtime_error("Dataset must have a defined size.");
  }
  // 初始化索引
  indices_.reserve(dataset_->size());
  for (size_t i = 0; i < dataset_->size(); ++i) {
    indices_.push_back(i);
  }
  if (shuffle_) {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
  }
}