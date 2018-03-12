# C-ATTL3 [![Build Status](https://travis-ci.org/ViktorC/C-ATTL3.svg?branch=master)](https://travis-ci.org/ViktorC/C-ATTL3)
A header-only neural network template library written in C++. C-ATTL3 relies heavily on [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library. It allows for the easy construction and training of both feed-forward and recurrent neural networks ranging from simple MLPs and RNNs to state-of-the-art InceptionNets, ResNets, DenseNets, convolutional LSTMs, and other complex architectures. C-ATTL3 supports data samples of different ranks and different floating point scalar types such as `float`, `double`, and `long double`. The Doxygen documentation of the library can be found [here](https://viktorc.github.io/C-ATTL3/html/).

## Components
The following sub-sections describe the main components of the C-ATTL3 deep learning library. Knowledge of these components and their relations is required for the effective usage of the library.

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

Most layers can handle data of rank 1 to 3 with the exception of convolutional and pooling layers which only accept rank 3 data. The actual rank of the input and output of the layers is one greater than the nominal rank of the layers to allow for batch learning. In the case of a layer with a nominal rank of 3, the input tensor is expected to be a rank-4 tensor with its ranks representing the sample number, height, width, and depth/channel (N,H,W,C) respectively. The nominal dimensionalities of the accepted input tensors of the different layers are specified using instances of the [Dimensions](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dimensions.html) class which relies on expression templates and compile time polymorphism to enable the fast and easy computation of the input dimensions of intermediary layers in complex neural networks. Besides the input dimensions, the one parameter required by all, most layers rely on multiple other hyper-parameters as well (e.g. max-norm constraint, dilation, receptor field size, etc.). These parameters may need to be fine-tuned manually or via random search (or in some other way) to optimize the behaviour of the networks.

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

These neural networks are either sequential (S) or non-sequential (NS). Non-sequential networks handle inputs of rank 2 to 4 (one greater than their nominal ranks) where the first rank represents the samples thus allowing for (mini-) batch training. On the other hand, sequential networks handle inputs of rank 3 to 5 (two greater than their nominal ranks) where the first rank, similarly to that of non-sequential networks' inputs, represents the samples and the second rank represents the time steps. Feedforward neural networks are ordinary networks with a set of layers through which the non-sequential input is propagated. Parallel neural nets, on the other hand, contain one or more 'lanes' of non-sequential networks through which the input is simultaneously propagated and eventually concatenated along either the highest or lowest rank. Both vanilla recurrent neural networks and LSTMs support arbitrary output schedules and multiplicative integration; however, due to the fact that they are unrolled for propagation/backpropagation through time (no symbolic loop), their memory requirements can be fairly high depending on the number of time steps. A bidirectional network takes a unidirectional network which it clones and reverses yielding two recurrent networks processing the input data from its two opposite ends along the time step rank. The outputs of the two unidirectional subnets of a bidirectional net can be either concatenated or summed. Sequential networks function as wrappers around non-sequential networks allowing them to be used on sequential data by applying them to each time step. Composite neural networks, similarly to parallel networks, are composed of one or more sub-nets; however, these nets are stacked sequentially and can be either sequential or non-sequential. Residual networks and dense networks are implementations of the [ResNet](https://arxiv.org/abs/1512.03385) and [DenseNet](https://arxiv.org/abs/1608.06993) architectures that use non-sequential composite neural networks as their sub-modules.

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
Similarly to the layers, these optimizers rely on hyper-parameters as well. Besides the hyper-parameters, optimizers also require 'practically' differentiable loss functions and regularization penalty functions. The library provides the following out of the box loss functions:
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

Given these parameters, optimizers can be constructed and used for gradient checks (in case of using self-implemented sub-classes of the core interfaces) and network training. Both methods are parameterized by a neural network implementation and one or two data providers.

### DataProvider
Data providers are responsible for supplying the data used for gradient verification, training, and testing. Currently only an in-memory data provider implementation is provided by the library, but the addition of a general on-disk data provider and specialized providers for popular data sets such as MNIST, CIFAR, and ImageNet is planned.
* [DataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_data_provider.html) [A]
  * [InMemoryDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_in_memory_data_provider.html)

### Preprocessor
C-ATTL3 also contains two preporcessors that can be used to transform the input data. They are:
* [Preprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_preprocessor.html) [A]
  * [NormalizationPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_normalization_preprocessor.html)
  * [PCAPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_p_c_a_preprocessor.html)

## Usage
The following code snippets demonstrate the usage of the library via a simple example.

	using namespace cattle;
	TensorPtr<float,4> training_obs_ptr = TensorPtr<float,4>(new Tensor<float,4>(80, 32, 32, 3));
	TensorPtr<float,4> training_obj_ptr = TensorPtr<float,4>(new Tensor<float,4>(80, 1, 1, 1));
	training_obs_ptr->setRandom();
	training_obj_ptr->setRandom();
	InMemoryDataProvider<float,3,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));

To demonstrate the usage of the library's in-memory data provider, some random training data is generated. The training data is comprised of two tensors of rank 4 and type `float`; one for the observations and one for the objectives. The function to be approximated by the neural network is the mapping function between the observations and the objectives. The first rank of these tensors always denotes the samples and its value must be the same in the two tensors. In the example above, the training data consists of 80 observation-objective pairs. In case of sequential data, the second rank of the tensors denotes the time steps which can differ between the observations and the objectives (if the output sequence length of the network does not match the input sequence length); however, in this example, the tensors represent non-sequential data (see the third template argument of the data providers or the optimizer; or the fact that the network to be trained is an inherently non-sequential feed-forward neural network), thus the last 3 ranks describe the individual observation and objective instances. The nominal rank of the data here is thus 3; representing height, width, and depth. The observations are images with a resolution of 32x32 and 3 color channels, while the objectives are single scalars. The data is generated by filling the two tensors with random values between 0 and 1. Finally, the training data provider is created out of the two tensors by moving the two unique pointers referencing them to the `InMemoryDataProvider` constructor.

	TensorPtr<float,4> test_obs_ptr = TensorPtr<float,4>(new Tensor<float,4>(20, 32, 32, 3));
	TensorPtr<float,4> test_obj_ptr = TensorPtr<float,4>(new Tensor<float,4>(20, 1, 1, 1));
	test_obs_ptr->setRandom();
	test_obj_ptr->setRandom();
	InMemoryDataProvider<float,3,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));

The test data provider is created in the same way. This test data is used to assess the accuracy of the neural network on data it has not encountered during the training process. This provides a measure of the network's generalization ability; the difference between the network's accuracy on the training data and that on the test data is a metric of overfitting. The test data is usually a smaller portion of all the available data than the training data. In our example, it is 20 samples as opposed to the 80 comprising the training data. Note that all other ranks of the test observation and objective tensors must match those of the training observation and objective tensors.

	WeightInitSharedPtr<float> init(new HeWeightInitialization<float>());
	std::vector<LayerPtr<float,3>> layers(9);
	layers[0] = LayerPtr<float,3>(new ConvLayer<float>(training_prov.get_obs_dims(), 10, init, 5, 2));
	layers[1] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[0]->get_output_dims()));
	layers[2] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[1]->get_output_dims()));
	layers[3] = LayerPtr<float,3>(new ConvLayer<float>(layers[2]->get_output_dims(), 20, init));
	layers[4] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[3]->get_output_dims()));
	layers[5] = LayerPtr<float,3>(new MaxPoolingLayer<float>(layers[4]->get_output_dims()));
	layers[6] = LayerPtr<float,3>(new FCLayer<float,3>(layers[5]->get_output_dims(), 500, init));
	layers[7] = LayerPtr<float,3>(new ReLUActivationLayer<float,3>(layers[6]->get_output_dims()));
	layers[8] = LayerPtr<float,3>(new FCLayer<float,3>(layers[7]->get_output_dims(), 1, init));
	FeedforwardNeuralNetwork<float,3> nn(std::move(layers));

The next step is the construction of the neural network. The above snippet demonstrates that of a simple convolutional neural network. The neural network implementation used is `FeedforwardNeuralNetwork` which takes a vector of unique layer pointers. Each layer in the vector must have the same input dimensions as the output dimensions of the preceding layer. Notice how the dimensions of the outputs of the layers do not need to be calculated manually; they can be simply retrieved using the `get_output_dims` method of the previous layers. The example network consists of convolutional, max pooling, rectified linear unit, and fully connected layers. Convolutional and fully connected layers require weight initialization; due to its well-known compatibility with ReLU activations, He weight initialization is a good choice in our situation. As the `WeightInitialization` class specifies a stateless interface, multiple layers can use the same implementation instance (this is the reason they take shared pointers). Similarly to the unique tensor pointer arguments of the data providers, the vector of unique layer pointers required by the network's constructor must be moved as well as unique smart pointers cannot be copied.

	nn.init();

Once the network is constructed, it is appropriate to initialize it. An unitialized network is in an undefined state. The initialization of the network entails the initialization of all its layers' parameters. Care must be taken not to unintentionally overwrite learned parameters by re-initializing the network.

	LossSharedPtr<float,3,false> loss(new QuadraticLoss<float,3,false>());
	RegPenSharedPtr<float> reg(new ElasticNetRegularizationPenalty<float>());
	NadamOptimizer<float,3,false> opt(loss, reg, 20);

Having set up the data providers and the network, it is time to specify the loss function, the regularization penalty, and the optimizer. For the sake of simplicity (concerning the data generation), the quadratic loss function is used in our example. Like `WeightInitialization`, both `Loss` and `RegularizationPenalty` define stateless interfaces; this is why they are wrapped in shared pointers and why single instances can be used by multiple optimizers. The optimizer used in our example is the `NadamOptimizer` which is generally a good first choice. Note the consistency of the template arguments; the data providers, the preprocessor, the neural network, the loss function, and the optimizer must all have the same scalar type, rank, and sequentiality (and the regularization penalty must have the same scalar type as well). As specified by the third argument of the optimizer's constructor, the batch size used for training and testing is 20. This means that both the training and the test data instances are processed in batches of 20. After the processing of each training batch, the parameters of the network's layers are updated. In our case, an epoch thus involves 4 parameter updates. It should be noted that most optimizers have several hyper-parameters that usually have reasonable default values and thus do not necessarily need to be specified.

	opt.optimize(nn, training_prov, test_prov, 500);
	
With everything set up and ready, the optimization can commence. The four non-optional paramaters of the `optimize` method are the neural network whose paramaters are to be optimized, the training data provider, the test data provider, and the number of epochs for which the optimization should go on. For our optimizer, these 500 epochs mean 2000 parameter updates alltogether. The `optimize` method is moderately verbose; for every epoch, it prints the training and test losses to the standard out stream. It also prints a warning message in case the test loss is greater than at the previous epoch.

	Tensor<float,4> input(5, 32, 32, 3);
	input.setRandom();
	Tensor<float,4> prediction = nn.infer(input);
	
The final code snippet demonstrates the usage of the trained neural network for inference. A random input tensor of the correct nominal input dimensions is generated and fed to the `infer` method which has the nueral network propagate the tensor through its layers and output its prediction. As seen above, inference is not restricted to single instances but can be performed on batches of data as well.

More examples of neural network constructs can be found [here](https://github.com/ViktorC/C-ATTL3/blob/master/test/test.cpp).

## TODO
Planned features include additional data providers, network serialization and de-serialization, GRU neural net, CTC loss, evolutionary and second order optimization algorithms, GPU acceleration via the use of cuBLAS and cuDNN, proper automated testing using CMake, and more extensive documentation including practical examples.
