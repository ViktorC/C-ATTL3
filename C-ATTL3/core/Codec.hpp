/*
 * Codec.hpp
 *
 *  Created on: 09.05.2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_CODEC_H_
#define C_ATTL3_CODEC_H_

#include <cstddef>
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

} /* namespace cattle */

#endif /* C_ATTL3_CODEC_H_ */
