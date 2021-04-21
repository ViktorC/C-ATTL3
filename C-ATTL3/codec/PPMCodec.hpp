/*
 * PPMCodec.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CODEC_PPMCODEC_H_
#define C_ATTL3_CODEC_PPMCODEC_H_

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "core/Codec.hpp"

namespace cattle {

/**
 * An enumeration for PPM format types.
 */
enum PPMFormatType { P2, P3, P5, P6 };

/**
 * A PPM image encoder-decoder.
 */
template <typename Scalar, PPMFormatType Type = P6>
class PPMCodec : public Codec<Scalar, 3> {
  static_assert(Type >= P2 && Type <= P6, "illegal ppm format type argument");
  static constexpr int MAX_SINGLE_BYTE_VAL = 255;
  static constexpr int MAX_DOUBLE_BYTE_VAL = 65535;
  static constexpr int MAX_LINE_LENGTH = 70;
  static constexpr int MAX_VAL_STRING_LENGTH = 5;
  static constexpr int BUFFER_SIZE = 3072;
  static constexpr bool GRAY_SCALE = Type == P2 || Type == P5;
  static constexpr bool BINARY = Type == P5 || Type == P6;

 public:
  PPMCodec() : type(resolve_type_string()) {}
  inline void encode(const Tensor<Scalar, 3>& data, const std::string& file_path) const {
    assert(data.dimension(0) > 0 && data.dimension(1) > 0 && data.dimension(2) == (GRAY_SCALE ? 1 : 3));
    std::ofstream file_stream(file_path, BINARY ? std::ios::binary : std::ios::out);
    assert(file_stream.is_open());
    Tensor<Scalar, 0> max_tensor = data.maximum();
    const int max_val = (int)max_tensor(0u);
    assert(max_val >= 0 && max_val <= MAX_DOUBLE_BYTE_VAL);
    const bool single_byte = max_val <= MAX_SINGLE_BYTE_VAL;
    std::string header = type + "\n" + std::to_string(data.dimension(1)) + " " + std::to_string(data.dimension(0)) +
                         "\n" + std::to_string(max_val) + "\n";
    file_stream.write(header.c_str(), header.length());
    int ind = 0;
    // For non-binary formats.
    int last_line_break = 0;
    unsigned char buffer[+BUFFER_SIZE];
    for (std::size_t i = 0; i < data.dimension(0); ++i) {
      for (std::size_t j = 0; j < data.dimension(1); ++j) {
        for (std::size_t k = 0; k < data.dimension(2); ++k) {
          int val = data(i, j, k);
          assert(val >= 0);
          if (BINARY) {
            if (ind == +BUFFER_SIZE) {
              file_stream.write(reinterpret_cast<char*>(buffer), ind);
              ind = 0;
            }
            // The buffer size is divisible by 2; no need to worry about buffer overflow.
            if (!single_byte) buffer[ind++] = (unsigned char)(val >> 8);
            buffer[ind++] = (unsigned char)val;
          } else {
            if (ind >= +BUFFER_SIZE - (MAX_VAL_STRING_LENGTH + 1)) {
              file_stream.write(reinterpret_cast<char*>(buffer), ind);
              last_line_break -= ind;
              ind = 0;
            }
            std::string val_string = std::to_string(val);
            for (int l = 0; l < val_string.length(); ++l) buffer[ind++] = *(val_string.c_str() + l);
            if ((ind + 1 - last_line_break >= (MAX_LINE_LENGTH - MAX_VAL_STRING_LENGTH))) {
              buffer[ind++] = '\n';
              last_line_break = ind;
            } else
              buffer[ind++] = ' ';
          }
        }
      }
    }
    if (ind != 0) file_stream.write(reinterpret_cast<char*>(buffer), ind);
  }
  inline Tensor<Scalar, 3> decode(const std::string& file_path) const {
    std::ifstream file_stream(file_path, BINARY ? std::ios::binary : std::ios::in);
    assert(file_stream.is_open());
    std::string format_type, dims, max_val_string;
    int width, height, max_val;
    std::getline(file_stream, format_type);
    assert(type == format_type);
    std::getline(file_stream, dims);
    std::istringstream dims_stream(dims);
    dims_stream >> width;
    dims_stream >> height;
    assert(width > 0 && height > 0);
    std::getline(file_stream, max_val_string);
    std::istringstream max_val_stream(max_val_string);
    max_val_stream >> max_val;
    assert(max_val >= 0 && max_val <= MAX_DOUBLE_BYTE_VAL);
    const std::size_t depth = GRAY_SCALE ? 1u : 3u;
    const int total_values = height * width * depth;
    Tensor<Scalar, 3> data((std::size_t)height, (std::size_t)width, depth);
    unsigned char buffer[+BUFFER_SIZE];
    int ind = 0;
    if (BINARY) {
      const bool single_byte = max_val <= MAX_SINGLE_BYTE_VAL;
      int values_in_buffer = std::min(+BUFFER_SIZE, (2 - single_byte) * total_values);
      int read_values = values_in_buffer;
      file_stream.read(reinterpret_cast<char*>(&buffer), values_in_buffer);
      assert(file_stream.gcount() == values_in_buffer);
      for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
          for (std::size_t k = 0; k < depth; ++k) {
            if (ind == values_in_buffer) {
              values_in_buffer = std::min(+BUFFER_SIZE, (2 - single_byte) * total_values - read_values);
              file_stream.read(reinterpret_cast<char*>(&buffer), values_in_buffer);
              assert(file_stream.gcount() == values_in_buffer);
              read_values += values_in_buffer;
              ind = 0;
            }
            unsigned val;
            if (single_byte)
              val = (unsigned)buffer[ind++];
            else {  // No buffer overflow possible due to the even buffer size.
              val = (unsigned)buffer[ind++];
              val |= ((unsigned)buffer[ind++]) << 8;
            }
            data(i, j, k) = (Scalar)val;
          }
        }
      }
    } else {
      file_stream.read(reinterpret_cast<char*>(&buffer), +BUFFER_SIZE);
      for (std::size_t i = 0; i < height; ++i) {
        for (std::size_t j = 0; j < width; ++j) {
          for (std::size_t k = 0; k < depth; ++k) {
            std::vector<unsigned char> chars;
            bool found_num = false;
            for (;;) {
              if (ind == +BUFFER_SIZE) {
                file_stream.read(reinterpret_cast<char*>(&buffer), +BUFFER_SIZE);
                ind = 0;
              } else if (ind == file_stream.gcount())
                break;
              unsigned char curr_char = buffer[ind++];
              if (curr_char >= '0' && curr_char <= '9') {
                chars.push_back(curr_char);
                found_num = true;
              } else if (found_num)
                break;
            }
            assert(found_num);
            std::string val_string(chars.begin(), chars.end());
            data(i, j, k) = (Scalar)std::stoi(val_string);
          }
        }
      }
    }
    return data;
  }

 private:
  /**
   * @return The string representation of the PPM format type.
   */
  inline static std::string resolve_type_string() {
    switch (Type) {
      case P2:
        return "P2";
      case P3:
        return "P3";
      case P5:
        return "P5";
      case P6:
        return "P6";
      default:
        return "";
    }
  }
  const std::string type;
};

} /* namespace cattle */

#endif /* C_ATTL3_CODEC_PPMCODEC_H_ */
