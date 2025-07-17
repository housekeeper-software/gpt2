#include "safe_tensors.h"
#include "json.hpp"
#include <iostream>
#include <optional>

namespace dense {
ModelParams::ModelParams() = default;
ModelParams::~ModelParams() = default;

bool ModelParams::load(const std::string &filename, ModelParams *params) {
  size_t file_size = 0;
  try {
    file_size = std::filesystem::file_size(filename);
  } catch (const std::filesystem::filesystem_error &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return false;
  }
  if (file_size < 8)
    return false;

  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    return false;
  }

  uint64_t header_size = 0;
  fread(&header_size, 1, sizeof(uint64_t), fp);
  if (file_size < header_size + 8) {
    fclose(fp);
    return false;
  }

  std::unique_ptr<char[]> header_buffer(new char[header_size + 1]);
  fread(header_buffer.get(), 1, header_size, fp);
  header_buffer[header_size] = 0;
  params->header_json = header_buffer.get();

  size_t data_size = file_size - header_size - 8;
  params->storage = std::make_shared<Storage>(data_size);
  fread(params->storage->data(), 1, data_size, fp);
  fclose(fp);

  try {
    nlohmann::json header_json = nlohmann::json::parse(header_buffer.get());

    for (const auto &[key, value] : header_json.items()) {
      if (key == "__metadata__") {
        for (const auto &[a, b] : value.items()) {
          if (b.is_string()) {
            params->meta_data.emplace(a, b.get<std::string>());
          }
        }
        continue;
      }

      std::optional<std::string> dtype;
      std::optional<std::vector<int64_t>> shape;
      std::optional<std::array<size_t, 2>> offset;

      for (const auto &[a, b] : value.items()) {
        if (a == "dtype") {
          dtype = b.get<std::string>();
        } else if (a == "shape") {
          shape = b.get<std::vector<int64_t>>();
        } else if (a == "data_offsets") {
          offset = b.get<std::array<size_t, 2>>();
        }
      }
      if (dtype && shape && offset && offset->size() > 1) {
        TensorInfo info;
        info.dtype = dtype.value();
        info.shape = shape.value();
        info.data_ptr = params->storage->data() + offset.value()[0];
        info.data_size = offset.value()[1] - offset.value()[0];
        params->tensors.emplace(key, std::move(info));
      } else {
        return false;
      }
    }
  } catch (const std::exception &e) {
    return false;
  }
  return true;
}

bool ModelParams::save(const std::string &filename) {
  remove(filename.c_str());

  if (meta_data.empty()) {
    meta_data.emplace("format", "pt");
  }
  nlohmann::json root;
  if (!meta_data.empty()) {
    nlohmann::json meta;
    for (const auto &i : meta_data) {
      meta[i.first] = i.second;
    }
    root["__metadata__"] = meta;
  }
  size_t current_offset = 0;
  for (const auto &i : tensors) {
    nlohmann::json info;
    info["dtype"] = i.second.dtype;
    info["shape"] = i.second.shape;
    std::array<size_t, 2> data_offset;
    data_offset[0] = current_offset;
    data_offset[1] = current_offset + i.second.data_size;
    current_offset += i.second.data_size;
    info["data_offsets"] = data_offset;
    root[i.first] = info;
  }

  header_json = root.dump();
  FILE *fp = fopen(filename.c_str(), "wb");
  uint64_t header_size = header_json.size();
  fwrite(&header_size, 1, sizeof(uint64_t), fp);
  fwrite(header_json.c_str(), 1, header_size, fp);

  for (const auto &i : tensors) {
    fwrite(i.second.data_ptr, 1, i.second.data_size, fp);
  }
  fclose(fp);
  return true;
}

} // namespace dense