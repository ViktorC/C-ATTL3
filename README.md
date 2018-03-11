# C-ATTL3
A neural network library written in C++. C-ATTL3 uses [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library. It allows for the easy construction and training of both feed-forward and recurrent neural networks ranging from simple MLPs and RNNs to state-of-the-art InceptionNets, ResNets, DenseNets, convolutional LSTMs, and other complex architecures. C-ATTL3 supports rank 1, 2, and 3 data and different floating point scalar types such as `float`, `double`, and `long double`. The Doxygen documentation of the library can be found [here](https://viktorc.github.io/C-ATTL3/html/).

The highest level building blocks of the different architectures are the neural network implementations provided by the library. Using these implementations, either as modules in a composite constellation or as standalone networks, almost any neural network architecture can be constructed. They are the following:
* FeedforwardNeuralNetwork (NS)
* ParallelNeuralNetwork (NS)
* UnidirectionalNeuralNetwork (S)
  * RecurrentNeuralNetwork (S)
  * LSTMNeuralNetwork (S)
* BidirectionalNeuralNetwork (S)
* SequentialNeuralNetwork (S)
* CompositeNeuralNetwork (NS/S)
* ResidualNeuralNetwork (NS)
* DenseNeuralNetwork (NS)

These neural networks are either sequential (S) or non-sequential (NS). Non-sequential networks handle inputs of rank 2 to 4 where the first rank represents the samples thus allowing for (mini-) batch training. On the other hand, sequential networks handle inputs of rank 3 to 5 where the first rank, similarly to that of non-sequential networks' inputs, represents the samples and the second rank represents the time steps. Feedforward neural networks are ordinary networks with a set of layers through which the non-sequential input is propagated. Parallel neural nets, on the other hand, contain one or more 'lanes' of non-sequential networks through which the input is simultaneously propagated and eventually concatenated along either the highest or lowest rank. Both vanilla recurrent neural networks and LSTMs support arbitrary output schedules and multiplicative integration. A bidirectional network takes a unidirectional network which it clones and reverses yielding two recurrent networks processing the input data from its two opposite ends along the time step rank. The outputs of the two unidirectional subnets of a bidirectional net can be either concatenated or summed. Sequential networks function as wrappers around non-sequential networks allowing them to be used on sequential data by applying them to each time step. Composite neural networks, similarly to parallel networks, are composed of one or more sub-nets; however, these nets are stacked sequentially and can be either sequential or non-sequential. Residual networks and dense networks are implementations of the [ResNet](https://arxiv.org/abs/1512.03385) and [DenseNet](https://arxiv.org/abs/1608.06993) architectures that use non-sequential composite neural networks as their sub-modules.

The lower level building blocks of neural networks are the layers. C-ATTL3 contains a wide selection of layers that can be used for the construction of highly effective sequential neural network modules. The available layer types are the following:
* KernelLayer
  * FCLayer
  * ConvLayer (3)
* ActivationLayer
  * IdentityActivationLayer
  * ScalingActivationLayer
  * BinaryStepActivationLayer
  * SigmoidActivationLayer
  * TanhActivationLayer
  * SoftmaxActivationLayer
  * ReLUActivationLayer
  * LeakyReLUActivationLayer
  * ELUActivationLayer
  * PReLUActivationLayer
* PoolingLayer (3)
  * SumPoolingLayer (3)
  * MaxPoolingLayer (3)
  * MeanPoolingLayer (3)
* DropoutLayer
* BatchNormLayer

Besides the input dimensions, the one parameter required by all, each layer uses multiple hyper-parameters (e.g. max-norm constraint, dilation, receptor field size, etc.). These parameters can be fine-tuned to optimize the behaviour of the networks. The fully-connected and convolutional layers also require weight initialization. The out-of-the-box weight initialization algorithms include:
* ZeroWeightInitialization
* OneWeightInitialization
* LeCunWeightInitialization
* GlorotWeightInitialization
* HeWeightInitialization
* OrthogonalWeightInitialization

The library also provides optimizers that can be used to train the networks via backpropagation. The currently available (first-order gradient descent) optimizers include the following:
* VanillaSGDOptimizer
* MomentumAcceleratedSGDOptimizer
* NesterovMomentumAcceleratedSGDOptimizer
* AdagradOptimizer
* RMSPropOptimizer
* AdadeltaOptimizer
* AdamOptimizer
* AdaMaxOptimizer
* NadamOptimizer

Similarly to the layers, these optimizers rely on hyper-parameters as well. Besides the hyper-parameters, optimizers also require more-or-less differentiable loss functions and regularization penalty functions. The library provides the following out of the box loss functions:
* QuadraticLoss
* HingeLoss
* CrossEntropyLoss
* MultiLabelHingeLoss
* MultiLabelLogLoss

The standard regularization penalties are:
* NoRegularizationPenalty
* L1RegularizationPenalty
* L2RegularizationPenalty
* ElasticNetRegularizationPenalty

Given these parameters, optimizers can be constructed and used for gradient checks (in case of using self-implemented sub-classes of the core interfaces) and network training. Both methods are parameterized by a neural network implementation and one or two data providers. Data providers are responsible for supplying the data used for gradient verification, training, and testing. Currently only an in-memory data provider implementation is provided by the library, but the addition of a general on-disk data provider and specialized providers for popular data sets such as MNIST, CIFAR, and ImageNet is planned.

C-ATTL3 also contains two preporcessors that can be used to transform the input data. They are:
* NormalizationPreprocessor
* PCAPreprocessor

Once a neural network has been trained, it can be used for inference effortlessly. The following code snippet demonstrates the usage of the library via a simple example.

	using namespace cattle;
	
	// Generate random training data
	TensorPtr<Scalar,4> training_obs_ptr = TensorPtr<Scalar,4>(new Tensor<Scalar,4>(80, 32, 32, 3));
	TensorPtr<Scalar,4> training_obj_ptr = TensorPtr<Scalar,4>(new Tensor<Scalar,4>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,3,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));

	// Generate random test data
	TensorPtr<Scalar,4> test_obs_ptr = TensorPtr<Scalar,4>(new Tensor<Scalar,4>(20, 32, 32, 3));
	TensorPtr<Scalar,4> test_obj_ptr = TensorPtr<Scalar,4>(new Tensor<Scalar,4>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar,3,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));

	// Construct a simple convolutional neural network.
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<LayerPtr<Scalar,3>> layers(9);
	layers[0] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 10, init, 5, 2));
	layers[1] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar,3>(new MaxPoolingLayer<Scalar>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<Scalar,3>(new ConvLayer<Scalar>(layers[2]->get_output_dims(), 20, init));
	layers[4] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<Scalar,3>(new MaxPoolingLayer<Scalar>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[5]->get_output_dims(), 500, init));
	layers[7] = LayerPtr<Scalar,3>(new ReLUActivationLayer<Scalar,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<Scalar,3>(new FCLayer<Scalar,3>(layers[7]->get_output_dims(), 1, init));
	FeedforwardNeuralNetwork<Scalar,3> nn(std::move(layers));

	// Initialize the network.
	nn.init();

	// Construct the optimizer.
	LossSharedPtr<Scalar,3,false> loss(new QuadraticLoss<Scalar,3,false>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar,3,false> opt(loss, reg, 20);

	// Train the network for 500 epochs.
	opt.optimize(nn, training_prov, test_prov, 500);
	
	// Generate random input data.
	Tensor<Scalar,4> input(5, 32, 32, 3);
	input.setRandom();
	
	// Inference
	Tensor<Scalar,4> prediction = nn.infer(input);

Planned features include additional data providers, network serialization and de-serialization, GRU network, CTC loss, evolutionary and second order optimization algorithms, and GPU support via the use of cuBLAS and cuDNN.