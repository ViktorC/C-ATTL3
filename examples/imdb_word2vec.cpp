/*
 * imdb_word2vec.cpp
 *
 *  Created on: 6 Jul 2018
 *      Author: Viktor Csomor
 */

#include "Cattle.hpp"

int main() {
	using namespace cattle;
	// Set up the IMDB data set provider.
	std::string imdb_folder = "data/imdb/";
	auto vocab = IMDBDataProvider<float>::build_vocab(imdb_folder + "imdb.vocab");
	const std::size_t vocab_size = vocab->size();
	IMDBDataProvider<float> train_prov(imdb_folder + "train/pos", imdb_folder + "train/neg", vocab);
	// Set up the word2vec neural network.
	const std::size_t latent_dims = 100;
	auto init = std::make_shared<GlorotParameterInitialization<float>>();
	std::vector<NeuralNetPtr<float,1,false>> modules(2);
	modules[0] = NeuralNetPtr<float,1,false>(new FeedforwardNeuralNetwork<float,1>(
			LayerPtr<float,1>(new DenseKernelLayer<float>({ vocab_size }, latent_dims, init))));
	modules[1] = NeuralNetPtr<float,1,false>(new FeedforwardNeuralNetwork<float,1>(
			LayerPtr<float,1>(new DenseKernelLayer<float>({ latent_dims }, vocab_size, init))));
	StackedNeuralNetwork<float,1,false> w2v(std::move(modules));
	w2v.init();
	// Create the loss and the optimizer.
	auto loss = std::make_shared<SoftmaxCrossEntropyLoss<float,1,false>>();
	NadamOptimizer<float,1,false> opt(loss, 100);
	opt.fit(w2v);
	// Optimize the network either as a CBOW model.
	const std::size_t epochs = 10;
	const std::size_t context_window = 3;
	std::array<std::size_t,2> offsets({ 0u, 0u });
	std::array<std::size_t,2> extents({ 1, vocab_size });
	for (std::size_t i = 0; i < epochs; ++i) {
		std::string epoch_header = "*   W2V Epoch " + std::to_string(i) + "   *";
		std::cout << std::endl << std::string(epoch_header.length(), '*') << std::endl <<
				"*" << std::string(epoch_header.length() - 2, ' ') << "*" << std::endl <<
				epoch_header << std::endl <<
				"*" << std::string(epoch_header.length() - 2, ' ') << "*" << std::endl <<
				std::string(epoch_header.length(), '*') << std::endl;
		double total_error = 0;
		std::size_t documents = 0;
		train_prov.reset();
		while (train_prov.has_more()) {
			auto doc_batch = train_prov.get_data(1).first;
			const std::size_t seq_length = doc_batch.dimension(1);
			MatrixMap<float> doc(doc_batch.data(), seq_length, vocab_size);
			TensorPtr<float,2> batch_obs(new Tensor<float,2>(seq_length, vocab_size));
			TensorPtr<float,2> batch_obj(new Tensor<float,2>(seq_length, vocab_size));
			bool train = true;
			for (std::size_t j = 0; j < seq_length; ++j) {
				Matrix<float> word = doc.row(j);
				Matrix<float> context;
				if (j > 0 && j < seq_length - 1) {
					std::size_t left_context_beg = (std::size_t) std::max(0, (int) j - (int) context_window);
					Matrix<float> left_context = doc.block(left_context_beg, 0u, j - left_context_beg, doc.cols());
					std::size_t right_context_end = std::min(seq_length - 1, j + context_window);
					Matrix<float> right_context = doc.block(j + 1, 0u, right_context_end - j, doc.cols());
					context = left_context.colwise().sum() * (1.0f / left_context.rows()) +
							right_context.colwise().sum() * (1.0f / right_context.rows());
				} else if (j > 0) {
					std::size_t left_context_beg = (std::size_t) std::max(0, (int) j - (int) context_window);
					Matrix<float> left_context = doc.block(left_context_beg, 0u, j - left_context_beg, doc.cols());
					context = left_context.colwise().sum() * (1.0f / left_context.rows());
				} else if (j < seq_length - 1) {
					std::size_t right_context_end = std::min(seq_length - 1, j + context_window);
					Matrix<float> right_context = doc.block(j + 1, 0u, right_context_end - j, doc.cols());
					context = right_context.colwise().sum() * (1.0f / right_context.rows());
				} else {
					train = false;
					break;
				}
				offsets[0] = j;
				batch_obs->slice(offsets, extents) = TensorMap<float,2>(context.data(), extents);
				batch_obj->slice(offsets, extents) = TensorMap<float,2>(word.data(), extents);
			}
			if (train) {
				doc_batch = Tensor<float,3>();
				MemoryDataProvider<float,1,false,false> data_prov(std::move(batch_obs), std::move(batch_obj));
				total_error += opt.train(w2v, data_prov, 1);
				++documents;
				std::cout << "\ttraining error: " << (total_error / documents) << std::endl;
			}
		}
	}
}
