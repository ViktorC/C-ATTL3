/*
 * Codec.hpp
 *
 *  Created on: 09.05.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_UTILS_CODEC_H_
#define C_ATTL3_UTILS_CODEC_H_

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "EigenProxy.hpp"

namespace cattle {

/**
 * An interface template for coder-decoders.
 */
template<typename Scalar, std::size_t Rank>
class Codec {
public:
	virtual ~Codec() = default;
	/**
	 * Encodes the tensor and writes it to a file.
	 *
	 * @param data The tensor whose contents are to be encoded.
	 * @param file_path The path to the file to which the encoded data should be written.
	 * If it does not exist, it will be created; if it does, the encoded data is appended
	 * to the contents of the file.
	 */
	virtual void encode(const Tensor<Scalar,Rank>& data, const std::string& file_path) const = 0;
	/**
	 * Decodes the contents of a file into a tensor.
	 *
	 * @param file_path The path to the file containing the encoded data.
	 * @return The decoded data in the form of a tensor.
	 */
	virtual Tensor<Scalar,Rank> decode(const std::string& file_path) const = 0;
};

/**
 * A template class for a PPM image encoder-decoder.
 */
template<typename Scalar>
class PPMCodec : public Codec<Scalar,3> {
	static constexpr char MAGIC_NUM[] = "P6";
	static constexpr unsigned MAX_SINGLE_BYTE_VAL = 255;
	static constexpr unsigned MAX_DOUBLE_BYTE_VAL = 65535;
	static constexpr int BUFFER_SIZE = 32 * 32 * 3;
public:
	inline void encode(const Tensor<Scalar,3>& data, const std::string& file_path) const {
		assert(data.dimension(0) > 0 && data.dimensions(1) > 0 && data.dimension(2) == 3);
		std::ofstream file_stream(file_path, Binary ? std::ios::binary : std::ios::out);
		assert(file_stream.is_open());
		int max_val = (int) data.maximum();
		assert(max_val <= MAX_DOUBLE_BYTE_VAL);
		bool single_byte = max_val <= MAX_SINGLE_BYTE_VAL;
		std::string header = std::string(MAGIC_NUM) + " " + data.dimension(1) + " " +
				data.dimension(0) + " " + max_val + " ";
		file_stream.write(header.cstr(), header.length());
		unsigned ind = 0;
		char buffer[BUFFER_SIZE];
		for (std::size_t i = 0; i < data.dimension(0); ++i) {
			for (std::size_t j = 0; j < data.dimension(1); ++j) {
				for (std::size_t k = 0; j < data.dimension(2); ++k) {
					if (ind == BUFFER_SIZE) {
						file_stream.write(buffer, BUFFER_SIZE);
						ind = 0;
					}
					buffer[ind++] = (char) data(i,j,k);
				}
			}
		}
		if (ind != 0)
			file_stream.write(buffer, ind);
	}
	inline Tensor<Scalar,3> decode(const std::string& file_path) const {
		// TODO
		return Tensor<Scalar,3>();
	}
};

} /* namespace cattle */

#endif /* C_ATTL3_UTILS_CODEC_H_ */
