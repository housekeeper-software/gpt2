#ifndef TIKTOKENIZER_H_
#define TIKTOKENIZER_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

class Pcre2Regex {
public:
  Pcre2Regex(const std::string &pattern,
             uint32_t options = PCRE2_UCP | PCRE2_UTF);
  ~Pcre2Regex();

  Pcre2Regex &operator=(Pcre2Regex &&other) noexcept;
  Pcre2Regex(Pcre2Regex &&other) noexcept;

  pcre2_code *compiled_pattern() const { return compiled_pattern_; }

  pcre2_match_data *match_data() const { return match_data_; }

private:
  void Destory();
  pcre2_code *compiled_pattern_; // 编译后的正则表达式
  pcre2_match_data *match_data_; // 匹配数据块
  Pcre2Regex(const Pcre2Regex &) = delete;
  Pcre2Regex &operator=(const Pcre2Regex &) = delete;
};

class TikTokenizer {
public:
  TikTokenizer();
  ~TikTokenizer();

  bool load_vocabulary(const std::string &vocab_path);
  bool load_merge_rules(const std::string &merges_path);

  bool add_special_tokens(const std::set<std::string> &allowed_special);

  std::vector<int> encode(const std::string &text,
                          bool add_special_tokens = true);

  std::string decode(const std::vector<int> &ids);

  uint32_t n_tokens() const {
    return id_to_bytes_.size() + id_to_special_tokens_.size();
  }

  std::unordered_multimap<int32_t, std::vector<int32_t>>
  process_breakers(const std::vector<std::string> &seq_breakers);

private:
  std::vector<uint8_t>
  decode_gpt2_mapped_string_to_bytes(const std::string &s) const;

  void tokenize_chunk(const std::string &chunk, std::vector<int> &encoded_ids);

  std::vector<std::vector<uint8_t>>
  apply_bpe_merges(std::vector<std::vector<uint8_t>> current_pieces);

  std::vector<int> find_all_best_bpe_pairs(
      const std::vector<std::vector<uint8_t>> &pieces) const;

  void get_overlapping_token_sequences(
      const std::string &str,
      std::unordered_multimap<int32_t, std::vector<int32_t>> &token_sequences,
      int max_tail_len = -1);

  std::vector<int> bytes_to_unicode_chars_;
  std::map<int, uint8_t> unicode_chars_to_bytes_;

  std::map<std::string, int> special_tokens_id_;
  std::map<int, std::string> id_to_special_tokens_;

  std::map<std::vector<uint8_t>, int> bytes_to_id_;
  std::map<int, std::vector<uint8_t>> id_to_bytes_;

  std::map<std::pair<std::vector<uint8_t>, std::vector<uint8_t>>, int>
      bpe_ranks_;

  std::unique_ptr<Pcre2Regex> gpt2_token_pattern_;
  std::unique_ptr<Pcre2Regex> special_token_pattern_;
};

#endif // TIKTOKENIZER_H_