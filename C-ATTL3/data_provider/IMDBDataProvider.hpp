/*
 * IMDBDataProvider.hpp
 *
 *  Created on: 21 Jul 2018
 *      Author: Viktor Csomor
 */

#ifndef C_ATTL3_DATA_PROVIDER_IMDBDATAPROVIDER_H_
#define C_ATTL3_DATA_PROVIDER_IMDBDATAPROVIDER_H_

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#include "JointFileDataProvider.hpp"

namespace cattle {

/**
 * An alias for a read-only dictionary mapping words to indices.
 */
typedef const std::map<std::string,std::size_t> Vocab;

/**
 * An alias for a shared pointer to a vocabulary.
 */
typedef std::shared_ptr<Vocab> VocabSharedPtr;

/**
 * An enumeration for the different objective types to use for the IMDB data set.
 */
enum IMDBObjType { BINARY, SMOOTH, CATEGORICAL };

/**
 * A data provider template for the IMDB Large Movie Review Dataset.
 *
 * \see http://ai.stanford.edu/~amaas/data/sentiment/
 */
template<typename Scalar, IMDBObjType ObjType = BINARY>
class IMDBDataProvider : public JointFileDataProvider<Scalar,1,true> {
	static_assert(ObjType >= BINARY && ObjType <= CATEGORICAL, "invalid IMDB objective type");
	typedef DataProvider<Scalar,1,true> Root;
	typedef JointFileDataProvider<Scalar,1,true> Base;
	typedef std::array<std::size_t,Root::DATA_RANK> RankwiseArray;
	static constexpr std::size_t PREALLOC_SEQ_LENGTH = 100;
public:
	static constexpr std::size_t PAD_IND = 0;
	static constexpr std::size_t UNK_IND = 1;
	/**
	 * @param pos_reviews_folder_path The path to the folder containing the positive reviews
	 * (without a trailing path separator).
	 * @param neg_reviews_folder_path The pathe to the folder containing the negative reveies
	 * (without a trailing path separator).
	 * @param vocab A shared pointer to the vocabulary to use.
	 * @param seq_length The sequence length to trim or pad the data to so as to enable batch
	 * training. If it is set to 0, no sequence trimming or padding is to be performed (which
	 * is likely to make batch training impossible).
	 */
	inline IMDBDataProvider(std::string pos_reviews_folder_path, std::string neg_reviews_folder_path,
			VocabSharedPtr vocab, std::size_t seq_length = 0) :
				Base::JointFileDataProvider({ vocab->size() }, { ObjType == CATEGORICAL ? 10 : 1 },
						resolve_review_files(pos_reviews_folder_path, neg_reviews_folder_path)),
				vocab(vocab),
				seq_length(seq_length) {
		Base::reset();
	}
	/**
	 * It populates a dictionary mapping words to indices given the path to a
	 * vocabulary file.
	 *
	 * @param vocab_path The path to the file listing all words appearing in the
	 * corpus.
	 * @return A map data structure representing the corpus' vocabulary.
	 */
	inline static VocabSharedPtr build_vocab(std::string vocab_path) {
		std::ifstream vocab_stream(vocab_path);
		assert(vocab_stream.is_open());
		std::map<std::string,std::size_t> vocab;
		// Reserve the first two indices for padding and unknown words.
		vocab.emplace(std::make_pair("<PAD>", +PAD_IND));
		vocab.emplace(std::make_pair("<UNK>", +UNK_IND));
		std::size_t index = 2;
		std::string word;
		while (vocab_stream >> word)
			vocab.emplace(std::make_pair(word, index++));
		return std::make_shared<const std::map<std::string,std::size_t>>(std::move(vocab));
	}
	/**
	 * Converts the embedded text into a string.
	 *
	 * @param obs A tensor representing the embedded text.
	 * @param vocab A shared pointer to the vocabulary.
	 * @return The text in the form of a string reconstructed from the provided
	 * tensor using the specified vocabulary.
	 */
	inline static std::string convert_to_text(const Tensor<Scalar,3>& obs, VocabSharedPtr vocab) {
		std::stringstream text_stream;
		std::string separator("");
		for (std::size_t i = 0; i < obs.dimension(0); ++i) {
			for (std::size_t j = 0; j < obs.dimension(1); ++j) {
				for (std::size_t k = 0; k < obs.dimension(2); ++k) {
					if (obs(i,j,k) == 1) {
						for (auto& entry : *vocab) {
							if (entry.second == k) {
								text_stream << separator << entry.first;
								separator = " ";
							}
						}
					}
				}
			}
		}
		return text_stream.str();
	}
protected:
	/**
	 * @param dir_path The path to the directory.
	 * @param file_names A vector to be populated by the paths to all the files
	 * contained in the directory.
	 */
	inline static void read_files_in_dir(std::string dir_path, std::vector<std::string>& file_names) {
		auto dir_ptr = opendir(dir_path.c_str());
		struct dirent* dir_ent_ptr;
		while ((dir_ent_ptr = readdir(dir_ptr)))
			file_names.push_back(dir_path + "/" + std::string(dir_ent_ptr->d_name));
		closedir(dir_ptr);
	}
	/**
	 * @param pos_reviews_folder_path The path to the directory containing
	 * the positive movie reviews.
	 * @param neg_reviews_folder_path The path to the directory containing
	 * the negative movie reviews.
	 * @return A randomly shuffled vector of the paths to all files contained
	 * in the directory.
	 */
	inline static std::vector<std::string> resolve_review_files(std::string pos_reviews_folder_path,
			std::string neg_reviews_folder_path) {
		std::vector<std::string> file_names;
		read_files_in_dir(pos_reviews_folder_path, file_names);
		read_files_in_dir(neg_reviews_folder_path, file_names);
		std::random_shuffle(file_names.begin(), file_names.end());
		return file_names;
	}
	/**
	 * It cleans the document by replacing all unsupported characters and words.
	 *
	 * @param document A string stream to the document to clean
	 */
	inline static void clean_document(std::stringstream& document) {
		std::string doc_string = document.str();
		std::transform(doc_string.begin(), doc_string.end(), doc_string.begin(),
				static_cast<int (*)(int)>(std::tolower));
		// Replace illegal character sequences by white spaces.
		static std::regex illegal_regex("(<br />)+|([^a-zA-Z-'!\?]+)");
		doc_string = std::regex_replace(doc_string, illegal_regex, " ");
		// Add a white space before the supported punctuation marks.
		static std::regex punct_regex("([!\?]{1})");
		doc_string = std::regex_replace(doc_string, punct_regex, " $1");
		document.str(doc_string);
	}
	inline DataPair<Scalar,1,true> _get_data(const std::string& file_name, std::ifstream& file_stream,
			std::size_t batch_size) {
		assert(batch_size > 0);
		const bool fixed_seq_length = seq_length != 0;
		typename Root::Data obs(1, (fixed_seq_length ? seq_length : +PREALLOC_SEQ_LENGTH), Base::obs_dims(0u));
		typename Root::Data obj(1, 1, Base::obj_dims(0u));
		obs.setZero();
		// Parse the rating from the name of the file.
		std::size_t last_under_score = file_name.find_last_of('_');
		std::size_t last_period = file_name.find_last_of('.');
		std::string rating_string = file_name.substr(last_under_score + 1,
				last_period - last_under_score - 1);
		unsigned rating = (unsigned) std::stoi(rating_string);
		switch (ObjType) {
			case BINARY:
				obj(0u,0u,0u) = (Scalar) (rating > 5);
				break;
			case SMOOTH:
				obj(0u,0u,0u) = ((Scalar) (rating - 1)) / 9;
				break;
			case CATEGORICAL:
				obj.setZero();
				obj(0u,0u,rating) = (Scalar) 1;
				break;
			default:
				assert(false);
		}
		// Read the document into a string so it can be pre-processed.
		std::stringstream doc_stream;
		doc_stream << file_stream.rdbuf();
		clean_document(doc_stream);
		// Tokenize the document.
		std::size_t time_step = 0;
		std::string word;
		while (doc_stream >> word && (time_step < seq_length || !fixed_seq_length)) {
			std::size_t ind;
			Vocab::const_iterator val = vocab->find(word);
			ind = (val != vocab->end()) ? val->second : +UNK_IND;
			if (!fixed_seq_length && time_step >= obs.dimension(1)) {
				typename Root::Data extra_obs(1, +PREALLOC_SEQ_LENGTH, Base::obs_dims(0u));
				extra_obs.setZero();
				obs = typename Root::Data(obs.concatenate(std::move(extra_obs), 1));
			}
			obs(0u,time_step++,ind) = (Scalar) 1;
		}
		if (fixed_seq_length) {
			for (; time_step < seq_length; ++time_step)
				obs(0u,time_step,+PAD_IND) = (Scalar) 1;
		} else {
			if (time_step < obs.dimension(1)) {
				RankwiseArray offsets({ 0u, 0u, 0u });
				RankwiseArray extents({ 1u, time_step, Base::obs_dims(0u) });
				obs = typename Root::Data(obs.slice(offsets, extents));
			}
		}
		return std::make_pair(std::move(obs), std::move(obj));
	}
	inline std::size_t _skip(std::ifstream& file_stream, std::size_t instances) {
		file_stream.seekg(0, std::ios::end);
		return instances - 1;
	}
private:
	const VocabSharedPtr vocab;
	const std::size_t seq_length;
};

}

#endif /* C_ATTL3_DATA_PROVIDER_IMDBDATAPROVIDER_H_ */
