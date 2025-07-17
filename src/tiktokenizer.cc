#include "tiktokenizer.h"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

const char *kGPT2_TOKEN_PATTERN =
    R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";

std::vector<int> bytes_to_unicode() {
  std::vector<int> bs(256);
  int n = 0;
  for (int i = 0; i < 256; ++i) {
    // 判断当前字节是否是可打印的ASCII字符、扩展ASCII字符 (Latin-1 Supplement)
    if ((i >= '!' /*0x21*/ &&
         i <= '~' /*0x7E*/) ||      // ASCII可打印字符，不包括空格
        (i >= 0xA1 && i <= 0xAC) || // Latin-1 Supplement 字符集的一部分
        (i >= 0xAE && i <= 0xFF)) { // Latin-1 Supplement 字符集的另一部分
      // ASCII码可打印字符从 0x20~0x7E,一共95个字符
      // (i >= '!' && i <= '~') 这是可打印字符，但是不包含 0x20(space),总计94个
      // unicode 从 161 到 255 是被称为 Latin-1 Supplement 字符集，有 96
      // 个字符，其中有两个是不可见的，分别是 0xA0 和 0xAD (i >=0xA1 && i <=
      // 0xAC),这里属于扩展ASCII码 (i >=0xAE && i <= 0xFF)
      // 如果是可打印字符，直接映射到其本身的Unicode码点,94 + 12 + 82 = 188
      bs[i] = i;
    } else {
      // 对于不可打印或控制字符，则映射到一个从 256 开始递增的新码点
      // 256 - 188 = 68
      bs[i] = 256 + n;
      n++;
    }
  }
  return bs;
}

std::vector<uint8_t> string_to_bytes(const std::string &s) {
  return std::vector<uint8_t>(s.begin(), s.end());
}

std::string bytes_to_string(const std::vector<uint8_t> &bytes) {
  return std::string(bytes.begin(), bytes.end());
}

// 完整的正则表达式特殊字符转义
std::string escape_regex_special_chars(const std::string &str) {
  std::string escaped;
  escaped.reserve(str.length() * 2);
  for (char c : str) {
    // 转义所有正则表达式特殊字符
    if (strchr("|.^*+?()[]{}\\$-", c)) {
      escaped += '\\'; // 如果是特殊字符，添加反斜杠进行转义
    }
    escaped += c;
  }
  return escaped;
}

} // namespace

Pcre2Regex::Pcre2Regex(const std::string &pattern, uint32_t options)
    : compiled_pattern_(nullptr), match_data_(nullptr) {
  int errorcode;          // 存储编译时的错误码
  PCRE2_SIZE erroroffset; // 存储编译时的错误偏移
                          // 编译正则表达式
  compiled_pattern_ = pcre2_compile((PCRE2_SPTR)pattern.c_str(), // 模式字符串
                                    PCRE2_ZERO_TERMINATED, // 模式以空字符结尾
                                    options,               // 编译选项
                                    &errorcode,            // 错误码输出
                                    &erroroffset,          // 错误偏移输出
                                    nullptr); // 编译上下文 (这里不需要)
  if (!compiled_pattern_) {
    // 如果编译失败
    PCRE2_UCHAR buffer[256];
    pcre2_get_error_message(errorcode, buffer, sizeof(buffer));
    std::cerr << "错误: PCRE2 正则表达式编译失败: " << buffer << " at offset "
              << erroroffset << std::endl;
  }
  if (compiled_pattern_) {
    // 从编译后的模式创建匹配数据块
    match_data_ =
        pcre2_match_data_create_from_pattern(compiled_pattern_, nullptr);
    if (!match_data_) {
      std::cerr << "错误: PCRE2 匹配数据块创建失败。" << std::endl;
      Destory();
    }
  }
}

Pcre2Regex::~Pcre2Regex() { Destory(); }

Pcre2Regex::Pcre2Regex(Pcre2Regex &&other) noexcept
    : compiled_pattern_(other.compiled_pattern_),
      match_data_(other.match_data_) {
  other.compiled_pattern_ = nullptr;
  other.match_data_ = nullptr;
}

void Pcre2Regex::Destory() {
  if (compiled_pattern_ != nullptr) {
    pcre2_code_free(compiled_pattern_);
    compiled_pattern_ = nullptr;
  }
  if (match_data_ != nullptr) {
    pcre2_match_data_free(match_data_);
    match_data_ = nullptr;
  }
}

Pcre2Regex &Pcre2Regex::operator=(Pcre2Regex &&other) noexcept {
  if (this != &other) {
    Destory();

    compiled_pattern_ = other.compiled_pattern_;
    match_data_ = other.match_data_;

    other.compiled_pattern_ = nullptr;
    other.match_data_ = nullptr;
  }
  return *this;
}

TikTokenizer::TikTokenizer() {
  // 初始化字节到Unicode字符的映射表
  bytes_to_unicode_chars_ = bytes_to_unicode();
  // 初始化Unicode字符到字节的映射表（反向映射）
  for (int i = 0; i < bytes_to_unicode_chars_.size(); ++i) {
    unicode_chars_to_bytes_[bytes_to_unicode_chars_[i]] =
        static_cast<uint8_t>(i);
  }
  // 初始化GPT-2分词模式的正则表达式
  gpt2_token_pattern_ = std::make_unique<Pcre2Regex>(kGPT2_TOKEN_PATTERN);
  if (!gpt2_token_pattern_->compiled_pattern()) {
    throw std::runtime_error("初始化 TikTokenizer 失败:正则表达式编译失败.");
  }
}

TikTokenizer::~TikTokenizer() = default;

bool TikTokenizer::add_special_tokens(
    const std::set<std::string> &allowed_special) {
  // 构建一个用于分割的正则表达式，形如 (special_token1|special_token2|...)
  // 按长度降序排序，确保最长匹配优先
  std::vector<std::string> sorted_special(allowed_special.begin(),
                                          allowed_special.end());
  std::sort(
      sorted_special.begin(), sorted_special.end(),
      [](const auto &a, const auto &b) { return a.length() > b.length(); });

  std::string special_pattern_str;
  for (const auto &token : sorted_special) {
    if (!special_pattern_str.empty()) {
      special_pattern_str += "|"; // 拼接多个特殊token，用或(|)连接
    }
    special_pattern_str +=
        escape_regex_special_chars(token); // 转义特殊字符并添加到模式字符串
  }

  if (special_pattern_str.empty()) {
    return false;
  }
  // 创建特殊token的正则表达式对象
  special_token_pattern_ = std::make_unique<Pcre2Regex>(special_pattern_str);
  return special_token_pattern_->compiled_pattern() != nullptr;
}

bool TikTokenizer::load_vocabulary(const std::string &vocab_path) {
  std::ifstream file(vocab_path);
  if (!file.is_open()) {
    std::cerr << "错误: 无法打开词表文件: " << vocab_path << std::endl;
    return false;
  }

  try {

    nlohmann::json vocab_json;
    file >> vocab_json;

    // 清空现有映射表
    bytes_to_id_.clear();
    id_to_bytes_.clear();
    special_tokens_id_.clear();
    id_to_special_tokens_.clear();

    // 遍历json中的每个条目
    for (const auto &[token, value] : vocab_json.items()) {
      // 获取token对应的ID
      int id = value.get<int>();
      // 检查token是否是特殊token（例如 "<|endoftext|>"）
      if (token.size() >= 5 && token.rfind("<|", 0) == 0 &&
          token.back() == '>') {
        special_tokens_id_[token] = id;    // 存储特殊token到ID的映射
        id_to_special_tokens_[id] = token; // 存储ID到特殊token的映射
      } else {
        // 如果不是特殊token，解码GPT-2映射的字符串到原始字节序列
        auto original_bytes_seq = decode_gpt2_mapped_string_to_bytes(token);
        bytes_to_id_[original_bytes_seq] = id;
        id_to_bytes_[id] = original_bytes_seq;
      }
    }
    std::cout << "成功加载词表: " << vocab_path << "，包含 "
              << bytes_to_id_.size() << " 个普通条目和 "
              << special_tokens_id_.size() << " 个特殊条目。" << std::endl;
    return true;
  } catch (const nlohmann::json::parse_error &e) {
    std::cerr << "错误: 解析词表文件失败: " << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "错误: 加载词表时发生未知错误: " << e.what() << std::endl;
  }
  return false;
}

bool TikTokenizer::load_merge_rules(const std::string &merges_path) {
  std::ifstream file(merges_path);
  if (!file.is_open()) {
    std::cerr << "错误: 无法打开合并规则文件: " << merges_path << std::endl;
    return false;
  }
  // 清空BPE合并规则映射表
  bpe_ranks_.clear();

  std::string line;
  int rank = 0; // 合并规则的优先级，rank越小优先级越高
  while (std::getline(file, line)) {
    if (line.find("#version:") != std::string::npos)
      continue; // 不能用#作为元数据的判断，因合并规则中也包含#开头的规则

    std::istringstream iss(line);
    std::string first_part_str, second_part_str;
    if (!(iss >> first_part_str >> second_part_str)) {
      std::cerr << "警告: 跳过无效合并规则行: '" << line << "'" << std::endl;
      continue;
    }

    try {
      // 解码合并规则中的两个部分到字节序列
      auto first_bytes = decode_gpt2_mapped_string_to_bytes(first_part_str);
      auto second_bytes = decode_gpt2_mapped_string_to_bytes(second_part_str);
      // 将合并规则及其rank存储到bpe_ranks_中
      bpe_ranks_[{first_bytes, second_bytes}] = rank++;
    } catch (const std::exception &e) {
      std::cerr << "警告: 处理合并规则时出错 '" << line << "': " << e.what()
                << std::endl;
      continue;
    }
  }
  std::cout << "成功加载合并规则: " << merges_path << "，包含 "
            << bpe_ranks_.size() << " 个规则。" << std::endl;
  return true;
}

// 将GPT-2映射的字符串（词汇表中存储的token）解码回原始字节序列
std::vector<uint8_t>
TikTokenizer::decode_gpt2_mapped_string_to_bytes(const std::string &s) const {
  std::vector<uint8_t> result_bytes;
  result_bytes.reserve(s.size());

  std::string::const_iterator it = s.begin(); // 迭代器遍历输入字符串
  while (it != s.end()) {
    int unicode_char_code = 0; // 存储解析出的Unicode码点
    uint8_t c = *it;           // 获取当前字节
    if (c < 0x80) {
      // 处理单字节UTF-8字符 (ASCII)
      unicode_char_code = c;
      ++it;
    } else if ((c & 0xE0) == 0xC0) {
      // 处理双字节UTF-8字符
      if (std::distance(it, s.end()) < 2)
        throw std::runtime_error("无效的UTF-8序列");
      unicode_char_code = ((c & 0x1F) << 6) | ((*(it + 1) & 0x3F));
      it += 2;
    } else if ((c & 0xF0) == 0xE0) {
      // 处理三字节UTF-8字符
      if (std::distance(it, s.end()) < 3)
        throw std::runtime_error("无效的UTF-8序列");
      unicode_char_code =
          ((c & 0x0F) << 12) | ((*(it + 1) & 0x3F) << 6) | ((*(it + 2) & 0x3F));
      it += 3;
    } else if ((c & 0xF8) == 0xF0) {
      // 处理四字节UTF-8字符
      if (std::distance(it, s.end()) < 4)
        throw std::runtime_error("无效的UTF-8序列");
      unicode_char_code = ((c & 0x07) << 18) | ((*(it + 1) & 0x3F) << 12) |
                          ((*(it + 2) & 0x3F) << 6) | ((*(it + 3) & 0x3F));
      it += 4;
    } else {
      throw std::runtime_error("无效的UTF-8起始字节");
    }
    // 将解析出的Unicode码点映射回原始字节
    // 如果这些文件中出现了其他任意的 Unicode
    // 字符，它们将无法被正确地映射回原始字节
    auto map_it = unicode_chars_to_bytes_.find(unicode_char_code);
    if (map_it != unicode_chars_to_bytes_.end()) {
      result_bytes.push_back(map_it->second);
    } else {
      std::cerr << "警告: 发现未知的GPT-2映射字符 (Unicode码点: "
                << unicode_char_code << ")。跳过此字符。" << std::endl;
    }
  }
  return result_bytes;
}

// 对文本块进行分词（不包含特殊token）
void TikTokenizer::tokenize_chunk(const std::string &chunk,
                                  std::vector<int> &encoded_ids) {
  if (chunk.empty()) {
    return;
  }
  PCRE2_SPTR subject = (PCRE2_SPTR)chunk.c_str(); // 待匹配的文本
  PCRE2_SIZE subject_length = chunk.length();     // 文本长度
  PCRE2_SIZE offset = 0;                          // 当前匹配的起始偏移

  while (offset < subject_length) { // 循环直到文本末尾
                                    // 执行正则表达式匹配
    int rc = pcre2_match(gpt2_token_pattern_->compiled_pattern(), subject,
                         subject_length, offset, 0,
                         gpt2_token_pattern_->match_data(), nullptr);

    if (rc < 0) {
      // 如果匹配失败
      if (rc == PCRE2_ERROR_NOMATCH) {
        std::cerr << "警告: 普通文本块内存在无法匹配的子片段: '"
                  << chunk.substr(offset) << "'" << std::endl;
      }
      break;
    }

    // 获取匹配到的子串的起始和结束偏移
    PCRE2_SIZE *ovector =
        pcre2_get_ovector_pointer(gpt2_token_pattern_->match_data());
    PCRE2_SIZE match_start = ovector[0];
    PCRE2_SIZE match_end = ovector[1];

    // 提取匹配到的单词块
    std::string word_chunk = chunk.substr(match_start, match_end - match_start);
    // 将单词块转换为字节序列
    auto raw_word_bytes = string_to_bytes(word_chunk);

    std::vector<std::vector<uint8_t>> current_bpe_pieces;
    for (uint8_t b : raw_word_bytes) {
      // 将单词块的每个字节视为一个独立的BPE片段
      current_bpe_pieces.push_back({b});
    }

    // 应用BPE合并规则
    auto final_merged_pieces = apply_bpe_merges(current_bpe_pieces);

    // 将最终合并的BPE片段编码为ID
    for (const auto &piece_bytes : final_merged_pieces) {
      // 在字节到ID的映射中查找
      auto it = bytes_to_id_.find(piece_bytes);
      if (it != bytes_to_id_.end()) {
        encoded_ids.push_back(it->second);
      } else {
        std::cerr << "警告: 未知字节序列（未在词表中找到）: '"
                  << bytes_to_string(piece_bytes) << "'。跳过此部分。"
                  << std::endl;
      }
    }
    offset = match_end; // 更新偏移量到当前匹配的结束位置
  }
}

// 找到所有具有最佳rank的BPE合并配对的位置
std::vector<int> TikTokenizer::find_all_best_bpe_pairs(
    const std::vector<std::vector<uint8_t>> &pieces) const {
  int best_rank = -1; // 最佳合并的rank，-1表示未找到

  std::pair<std::vector<uint8_t>, std::vector<uint8_t>>
      best_pair; // 最佳合并的配对

  // 第一遍：找到最佳rank
  for (size_t i = 0; i < pieces.size() - 1; ++i) {
    // 遍历所有相邻的片段对
    const auto &p1 = pieces[i];          // 第一个片段
    const auto &p2 = pieces[i + 1];      // 第二个片段
    auto it = bpe_ranks_.find({p1, p2}); // 在bpe_ranks_中查找这对片段
    if (it != bpe_ranks_.end()) {
      // 如果是第一个找到的配对或rank更小
      if (best_rank == -1 || it->second < best_rank) {
        best_rank = it->second; // 更新最佳rank
        best_pair = {p1, p2};   // 更新最佳配对
      }
    }
  }

  if (best_rank == -1) // 如果没有找到任何可合并的配对
    return {};

  // 第二遍：找到所有具有最佳rank的位置
  std::vector<int> positions;
  for (size_t i = 0; i < pieces.size() - 1; ++i) {
    // 检查当前片段对是否与最佳配对相同
    if (pieces[i] == best_pair.first && pieces[i + 1] == best_pair.second) {
      positions.push_back(static_cast<int>(i)); // 记录位置
    }
  }
  return positions; // 返回所有最佳合并位置
}

// 应用BPE合并规则
std::vector<std::vector<uint8_t>> TikTokenizer::apply_bpe_merges(
    std::vector<std::vector<uint8_t>> current_pieces) {
  while (true) { // 循环直到没有更多的合并
    // 找到所有最佳合并位置
    auto positions = find_all_best_bpe_pairs(current_pieces);
    if (positions.empty())
      break;

    // 创建合并后的piece
    // (只根据第一个最佳位置的pair来创建，因为所有最佳pair会合并成相同的token)
    std::vector<uint8_t> merged_piece = current_pieces[positions[0]];
    merged_piece.insert(merged_piece.end(),
                        current_pieces[positions[0] + 1].begin(),
                        current_pieces[positions[0] + 1].end());

    // 存储合并后的新片段序列
    std::vector<std::vector<uint8_t>> new_pieces;
    int skip_next = 0; // 用于跳过已被合并的下一个片段

    for (size_t i = 0; i < current_pieces.size(); ++i) { // 遍历当前片段序列
      if (skip_next > 0) {
        // 如果需要跳过当前片段 (因为它已经被前一个片段合并)
        skip_next--;
        continue;
      }

      // 检查当前位置是否是需要合并的位置
      bool should_merge = false;
      for (int pos : positions) {
        if (static_cast<int>(i) == pos) {
          should_merge = true;
          break;
        }
      }

      if (should_merge && i + 1 < current_pieces.size()) {
        // 如果当前位置需要合并并且有下一个片段
        // 添加合并后的片段
        new_pieces.push_back(merged_piece);
        skip_next = 1; // 跳过下一个piece
      } else {
        // 否则，添加当前片段
        new_pieces.push_back(current_pieces[i]);
      }
    }
    // 更新当前片段序列为新序列
    current_pieces = std::move(new_pieces);
  }
  // 返回最终合并后的片段序列
  return current_pieces;
}

// 将文本编码为token ID序列
std::vector<int> TikTokenizer::encode(const std::string &text,
                                      bool add_special_tokens) {
  std::vector<int> encoded_ids; // 存储编码后的ID

  // 1. 如果不允许添加特殊token，则将整个文本视为一个普通块进行分词
  if (!add_special_tokens) {
    tokenize_chunk(text, encoded_ids);
    return encoded_ids;
  }

  // 如果没有设置特殊token的正则表达式，也直接作为普通文本块处理
  if (!special_token_pattern_->compiled_pattern()) {
    tokenize_chunk(text, encoded_ids);
    return encoded_ids;
  }

  // 2. 使用特殊token正则表达式来分割文本
  size_t last_pos = 0;                           // 上一个匹配结束的位置
  PCRE2_SIZE offset = 0;                         // 当前匹配的起始偏移
  PCRE2_SPTR subject = (PCRE2_SPTR)text.c_str(); // 待匹配的文本
  PCRE2_SIZE subject_length = text.length();     // 文本长度

  while (offset <= subject_length) { // 循环直到文本末尾
                                     // 执行特殊token的正则表达式匹配
    int rc = pcre2_match(special_token_pattern_->compiled_pattern(), subject,
                         subject_length, offset, 0,
                         special_token_pattern_->match_data(), nullptr);
    if (rc < 0) {
      // 没有更多匹配，退出循环
      break;
    }

    // 获取匹配到的特殊token的起始和结束偏移
    PCRE2_SIZE *ovector =
        pcre2_get_ovector_pointer(special_token_pattern_->match_data());
    size_t start = ovector[0];
    size_t end = ovector[1];

    // 2a. 处理匹配到的特殊token之前的【普通文本块】
    if (start > last_pos) {
      tokenize_chunk(text.substr(last_pos, start - last_pos), encoded_ids);
    }

    // 2b. 处理【特殊token】本身
    auto special_token = text.substr(start, end - start);
    auto it = special_tokens_id_.find(special_token); // 在特殊token映射中查找
    if (it != special_tokens_id_.end()) {
      encoded_ids.push_back(it->second); // 如果找到，添加到编码ID列表
    } else {
      std::cerr << "警告: 特殊token '" << special_token << "' 未在词表中找到。"
                << std::endl;
    }

    last_pos = end; // 更新上一个匹配结束的位置
    offset = end;   // 更新当前匹配的起始偏移

    // 防止无限循环：如果匹配了零长度（PCRE2可能返回零长度匹配，但实际文本没有消耗），强制前进
    if (start == end) {
      offset++;
    }
  }

  // 3. 处理最后一个特殊token之后的【普通文本块】
  if (last_pos < text.length()) {
    tokenize_chunk(text.substr(last_pos), encoded_ids);
  }

  return encoded_ids; // 返回最终编码ID序列
}

// 将token ID序列解码回字符串
std::string TikTokenizer::decode(const std::vector<int> &ids) {
  std::vector<uint8_t> all_raw_bytes;    // 存储所有解码后的原始字节
  all_raw_bytes.reserve(ids.size() * 4); // 预分配空间，估算每个ID平均4个字节

  for (auto id : ids) {
    auto special_it =
        id_to_special_tokens_.find(id); // 尝试在特殊token映射中查找
    if (special_it != id_to_special_tokens_.end()) {
      // 将特殊token字符串转换为字节序列
      auto special_token_bytes = string_to_bytes(special_it->second);
      all_raw_bytes.insert(all_raw_bytes.end(), special_token_bytes.begin(),
                           special_token_bytes.end()); // 添加到结果字节序列
    } else {
      // 在普通token映射中查找
      auto it = id_to_bytes_.find(id);
      if (it != id_to_bytes_.end()) {
        all_raw_bytes.insert(all_raw_bytes.end(), it->second.begin(),
                             it->second.end()); // 添加到结果字节序列
      } else {
        std::cerr << "警告: 解码时未找到ID " << id
                  << " 对应的字节序列或特殊token。跳过此ID。" << std::endl;
      }
    }
  }
  // 将所有原始字节转换为字符串并返回
  return std::string(all_raw_bytes.begin(), all_raw_bytes.end());
}

// Ported from Koboldcpp, original PR:
// https://github.com/LostRuins/koboldcpp/pull/982 (Original author: pi6am)
void TikTokenizer::get_overlapping_token_sequences(
    const std::string &str,
    std::unordered_multimap<int32_t, std::vector<int32_t>> &token_sequences,
    int max_tail_len) {
  // 遍历所有可能的token ID
  for (int32_t token_id = 0; token_id < (int32_t)n_tokens(); token_id++) {
    // 解码当前token ID对应的字符串
    std::string word = decode({token_id});
    if (word.find(str) != std::string::npos) {
      // 如果解码后的word直接包含完整的str,添加到结果中，尾部序列为空
      token_sequences.emplace(token_id, std::vector<int32_t>());
    } else {
      size_t word_len = word.size();
      size_t str_len = str.size();
      size_t pos = -1;
      while ((pos = word.find(str[0], pos + 1)) != std::string::npos) {
        // 查找str的第一个字符在word中的所有出现位置
        bool match = true;
        size_t i;
        for (i = 1; i < str_len && i + pos < word_len;
             ++i) { // 从第二个字符开始比较
          if (word[pos + i] != str[i]) {
            // 如果字符不匹配
            match = false;
            break;
          }
        }
        if (match) { // 如果成功匹配了str的前缀
          // 对str中未匹配的部分进行编码 (tail_len)
          std::vector<int32_t> tokenization = encode(str.substr(i), false);
          if (max_tail_len >= 0 && tokenization.size() > (size_t)max_tail_len) {
            // 如果尾部序列长度超过限制,截断
            tokenization.resize(max_tail_len);
          }

          // 确保我们没有已经存在的重复匹配token化
          // 获取所有与当前token_id关联的序列
          auto its = token_sequences.equal_range(token_id);
          bool found = false;
          for (auto it = its.first; it != its.second; ++it) {
            if (tokenization == it->second) {
              // 如果找到相同的尾部序列
              found = true;
              break;
            }
          }
          if (!found) {
            // 如果是新的组合,添加到结果中
            token_sequences.emplace(token_id, tokenization);
          }
        }
      }
    }
  }
}

// 处理序列分隔符
std::unordered_multimap<int32_t, std::vector<int32_t>>
TikTokenizer::process_breakers(const std::vector<std::string> &seq_breakers) {
  const int MAX_CHAR_LEN = 40;
  const int MAX_SEQ_LEN = 20;
  std::unordered_multimap<int32_t, std::vector<int32_t>> processed_breakers;
  // 处理序列分隔符
  for (size_t i = 0; i < seq_breakers.size(); ++i) {
    if (seq_breakers[i].empty()) {
      std::cerr << "skipping null or empty DRY sequence breaker at index " << i
                << std::endl;
      continue;
    }

    std::string sequence_break(seq_breakers[i]);

    if (sequence_break.size() > MAX_CHAR_LEN) {
      std::cerr << "truncating DRY sequence breaker to " << MAX_CHAR_LEN
                << " characters\n"
                << std::endl;
      sequence_break.resize(MAX_CHAR_LEN);
    }
    // 获取与当前分隔符重叠的token序列
    get_overlapping_token_sequences(sequence_break, processed_breakers,
                                    MAX_SEQ_LEN);
  }
  return processed_breakers;
}