# C-ATTL3
A neural network library written in C++. C-ATTL3 uses [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library when run on the CPU. It allows for the easy construction and training of feed-forward neural networks ranging from simple MLPs to state-of-the-art convolutional InceptionNets, ResNets, Inception-ResNets, and DenseNets. C-ATTL3 also supports different floating point scalar types such as `float`, `double`, and `long double`.

The highest level building blocks of the different architectures are the neural network implementations provided by the library. Using these implementations, either as modules in a composite constellation or as standalone networks, almost any neural network architecture can be constructed. They are the following:
* SequentialNeuralNetwork
* ParallelNeuralNetwork
* CompositeNeuralNetwork
* ResidualNeuralNetwork
* DenseNeuralNetwork

Sequential neural networks are ordinary networks with a set of layers through which the input is propagated. Parallel neural nets, on the other hand, contain one or more 'lanes' of networks through which the input is simultaneously propagated and eventually concatenated along the depth dimension. Composite neural networks, similarly to parallel networks, are composed of one or more sub-nets; however, these nets are stacked sequentially. Residual networks and dense networks are implementations of the [ResNet](https://arxiv.org/abs/1512.03385) and [DenseNet](https://arxiv.org/abs/1608.06993) architectures that use composite neural networks as their sub-modules. Both the inputs and outputs of these neural networks are rank-four tensors which allow for the representation of batches of images or data of lower rank.

The lower level building blocks of neural networks are the layers. C-ATTL3 contains a wide selection of layers that can be used for the construction of highly effective sequential neural network modules. The available layer types are the following:
* FCLayer
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
* ConvLayer
* PoolingLayer
  * SumPoolingLayer
  * MaxPoolingLayer
  * MeanPoolingLayer
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
	Tensor4Ptr<Scalar> training_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 32, 32, 3));
	Tensor4Ptr<Scalar> training_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));

	// Generate random test data
	Tensor4Ptr<Scalar> test_obs_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 32, 32, 3));
	Tensor4Ptr<Scalar> test_obj_ptr = Tensor4Ptr<Scalar>(new Tensor4<Scalar>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<Scalar> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));

	// Construct a simple convolutional neural network.
	WeightInitSharedPtr<Scalar> init(new HeWeightInitialization<Scalar>());
	std::vector<LayerPtr<Scalar>> layers(9);
	layers[0] = LayerPtr<Scalar>(new ConvLayer<Scalar>(training_prov.get_obs_dims(), 10, init, 5, 2));
	layers[1] = LayerPtr<Scalar>(new ReLUActivationLayer<Scalar>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<Scalar>(new ConvLayer<Scalar>(layers[2]->get_output_dims(), 20, init));
	layers[4] = LayerPtr<Scalar>(new ReLUActivationLayer<Scalar>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<Scalar>(new MaxPoolingLayer<Scalar>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers[5]->get_output_dims(), 500, init));
	layers[7] = LayerPtr<Scalar>(new ReLUActivationLayer<Scalar>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<Scalar>(new FCLayer<Scalar>(layers[7]->get_output_dims(), 1, init));
	SequentialNeuralNetwork<Scalar> nn(std::move(layers));

	// Initialize the network.
	nn.init();

	// Construct the optimizer.
	LossSharedPtr<Scalar> loss(new QuadraticLoss<Scalar>());
	RegPenSharedPtr<Scalar> reg(new ElasticNetRegularizationPenalty<Scalar>());
	NadamOptimizer<Scalar> opt(loss, reg, 20);

	// Train the network for 500 epochs.
	opt.optimize(nn, training_prov, test_prov, 500);
	
	// Generate random input data.
	Tensor4<Scalar> input(5, 32, 32, 3);
	input.setRandom();
	
	// Inference
	Tensor4<Scalar> prediction = nn.infer(input);

Planned features include additional data providers, network serialization and de-serialization, LSTM and GRU layers, evolutionary and second order optimization algorithms, and GPU support via the use of cuBLAS and cuDNN.