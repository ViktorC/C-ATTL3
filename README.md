# C-ATTL3 [![Build Status](https://travis-ci.org/ViktorC/C-ATTL3.svg?branch=master)](https://travis-ci.org/ViktorC/C-ATTL3)
A header-only neural network template library written in C++. C-ATTL3 uses [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library. It allows for the easy construction and training of both feed-forward and recurrent neural networks ranging from simple MLPs and RNNs to state-of-the-art InceptionNets, ResNets, DenseNets, convolutional LSTMs, and other complex architectures. C-ATTL3 supports rank 1, 2, and 3 data and different floating point scalar types such as `float`, `double`, and `long double`. The Doxygen documentation of the library can be found [here](https://viktorc.github.io/C-ATTL3/html/).

## Components
The following section describes the main components of the C-ATTL3 deep learning library.

### Layer
The lowest level building blocks of neural networks in C-ATTL3 are the layers. The library provides a wide selection of them that can be used for the construction of neural network modules. The available layer types are the following:
* [Layer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_layer.html) [A]
  * [KernelLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_kernel_layer.html) [A]
    * [FCLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_f_c_layer.html)
    * [ConvLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_conv_layer.html) (3)
  * [ActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_activation_layer.html) [A]
    * [IdentityActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_identity_activation_layer.html)
    * [ScalingActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_scaling_activation_layer.html)
    * [BinaryStepActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_binary_step_activation_layer.html)
    * [SigmoidActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sigmoid_activation_layer.html)
    * [TanhActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_tanh_activation_layer.html)
    * [SoftmaxActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_softmax_activation_layer.html)
    * [ReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_re_l_u_activation_layer.html)
    * [LeakyReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_leaky_re_l_u_activation_layer.html)
    * [ELUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_e_l_u_activation_layer.html)
    * [PReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_p_re_l_u_activation_layer.html)
  * [PoolingLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_pooling_layer.html) [A] (3)
    * [SumPoolingLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sum_pooling_layer.html) (3)
    * [MaxPoolingLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_max_pooling_layer.html) (3)
    * [MeanPoolingLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_mean_pooling_layer.html) (3)
  * [BatchNormLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_batch_norm_layer.html)
  * [DropoutLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dropout_layer.html)

Most layers can handle data of rank 1 to 3 with the exception of convolutional and pooling layers which only accept rank 3 data. The actual input and output of the layers is of a rank one greater than the nominal rank of the layers to allow for batch learning. Besides the input dimensions, the one parameter required by all, each layer uses multiple hyper-parameters (e.g. max-norm constraint, dilation, receptor field size, etc.). These parameters can be fine-tuned to optimize the behaviour of the networks.

#### WeightInitialization
The kernel layers (fully-connected and convolutional) also require weight initialization. The out-of-the-box weight initialization algorithms include:
* [WeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_weight_initialization.html) [A]
  * [ZeroWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_zero_weight_initialization.html)
  * [OneWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_one_weight_initialization.html)
  * [GaussianWeightInitialization]() [A]
    * [LeCunWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_le_cun_weight_initialization.html)
    * [GlorotWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_glorot_weight_initialization.html)
    * [HeWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_he_weight_initialization.html)
    * [OrthogonalWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_orthogonal_weight_initialization.html)

The dimensionalities of the accepted input tensors of the different layers are specified using instances of the [Dimensions](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dimensions.html) class which uses expression templates and compile time polymorphism to enable the fast and easy computation of the input dimensionalities of intermediary layers in complex neural networks.

### NeuralNetwork
The highest level building blocks of the different architectures are the neural network implementations provided by the library. Using these implementations, either as modules in a composite constellation or as standalone networks, almost any neural network architecture can be constructed. They are the following:
* [NeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_neural_network.html) [A] (NS/S)
  * [FeedforwardNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_feedforward_neural_network.html) (NS)
  * [ParallelNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_parallel_neural_network.html) (NS)
  * [UnidirectionalNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_unidirectional_neural_network.html) [A] (S)
    * [RecurrentNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_recurrent_neural_network.html) (S)
    * [LSTMNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l_s_t_m_neural_network.html) (S)
  * [BidirectionalNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_bidirectional_neural_network.html) (S)
  * [SequentialNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sequential_neural_network.html) (S)
  * [CompositeNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_composite_neural_network.html) (NS/S)
  * [ResidualNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_residual_neural_network.html) (NS)
  * [DenseNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dense_neural_network.html) (NS)

These neural networks are either sequential (S) or non-sequential (NS). Non-sequential networks handle inputs of rank 2 to 4 (one greater than their nominal ranks) where the first rank represents the samples thus allowing for (mini-) batch training. On the other hand, sequential networks handle inputs of rank 3 to 5 (two greater than their nominal ranks) where the first rank, similarly to that of non-sequential networks' inputs, represents the samples and the second rank represents the time steps. Feedforward neural networks are ordinary networks with a set of layers through which the non-sequential input is propagated. Parallel neural nets, on the other hand, contain one or more 'lanes' of non-sequential networks through which the input is simultaneously propagated and eventually concatenated along either the highest or lowest rank. Both vanilla recurrent neural networks and LSTMs support arbitrary output schedules and multiplicative integration. A bidirectional network takes a unidirectional network which it clones and reverses yielding two recurrent networks processing the input data from its two opposite ends along the time step rank. The outputs of the two unidirectional subnets of a bidirectional net can be either concatenated or summed. Sequential networks function as wrappers around non-sequential networks allowing them to be used on sequential data by applying them to each time step. Composite neural networks, similarly to parallel networks, are composed of one or more sub-nets; however, these nets are stacked sequentially and can be either sequential or non-sequential. Residual networks and dense networks are implementations of the [ResNet](https://arxiv.org/abs/1512.03385) and [DenseNet](https://arxiv.org/abs/1608.06993) architectures that use non-sequential composite neural networks as their sub-modules.

### Optimizer
The library also provides optimizers that can be used to train the networks via backpropagation. The currently available (first-order gradient descent) optimizers include the following:
* [Optimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_optimizer.html) [A]
  * [SGDOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_s_g_d_optimizer.html) [A]
    * [VanillaSGDOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_vanilla_s_g_d_optimizer.html)
    * [MomentumAcceleratedSGDOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_momentum_accelerated_s_g_d_optimizer.html)
      * [NesterovMomentumAcceleratedSGDOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_nesterov_momentum_accelerated_s_g_d_optimizer.html)
    * [AdagradOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_adagrad_optimizer.html)
      * [RMSPropOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_r_m_s_prop_optimizer.html)
    * [AdadeltaOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_adadelta_optimizer.html)
    * [AdamOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_adam_optimizer.html)
      * [AdaMaxOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_ada_max_optimizer.html)
      * [NadamOptimizer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_nadam_optimizer.html)

#### Loss
Similarly to the layers, these optimizers rely on hyper-parameters as well. Besides the hyper-parameters, optimizers also require more-or-less differentiable loss functions and regularization penalty functions. The library provides the following out of the box loss functions:
* [Loss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_loss.html) [A]
  * [QuadraticLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_quadratic_loss.html)
  * [HingeLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_hinge_loss.html)
  * [CrossEntropyLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_cross_entropy_loss.html)
  * [MultiLabelHingeLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_multi_label_hinge_loss.html)
  * [MultiLabelLogLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_multi_label_log_loss.html)

#### RegularizationPenalty
The standard regularization penalties are:
* [RegularizationPenalty](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_regularization_penalty.html) [A]
  * [NoRegularizationPenalty](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_no_regularization_penalty.html)
  * [L1RegularizationPenalty](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l1_regularization_penalty.html)
  * [L2RegularizationPenalty](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l2_regularization_penalty.html)
  * [ElasticNetRegularizationPenalty](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_elastic_net_regularization_penalty.html)

### DataProvider
Given these parameters, optimizers can be constructed and used for gradient checks (in case of using self-implemented sub-classes of the core interfaces) and network training. Both methods are parameterized by a neural network implementation and one or two data providers. Data providers are responsible for supplying the data used for gradient verification, training, and testing. Currently only an in-memory data provider implementation is provided by the library, but the addition of a general on-disk data provider and specialized providers for popular data sets such as MNIST, CIFAR, and ImageNet is planned.
* [DataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_data_provider.html) [A]
  * [InMemoryDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_in_memory_data_provider.html)

### Preprocessor
C-ATTL3 also contains two preporcessors that can be used to transform the input data. They are:
* [Preprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_preprocessor.html) [A]
  * [NormalizationPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_normalization_preprocessor.html)
  * [PCAPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_p_c_a_preprocessor.html)

## Usage
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

More examples of neural network specifications can be found [here](https://github.com/ViktorC/C-ATTL3/blob/master/test/test.cpp).

## TODO
Planned features include additional data providers, network serialization and de-serialization, GRU network, CTC loss, evolutionary and second order optimization algorithms, and GPU acceleration via the use of cuBLAS and cuDNN.