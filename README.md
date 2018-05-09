# C-ATTL3 [![Codecov](https://img.shields.io/codecov/c/github/ViktorC/C-ATTL3.svg)](https://codecov.io/gh/ViktorC/C-ATTL3)
| Clang++ 5.0 | G++ 7 |
|---|---|
| [![Clang](https://travis-matrix-badges.herokuapp.com/repos/ViktorC/C-ATTL3/branches/master/1)](https://travis-ci.org/ViktorC/C-ATTL3) | [![GCC](https://travis-matrix-badges.herokuapp.com/repos/ViktorC/C-ATTL3/branches/master/2)](https://travis-ci.org/ViktorC/C-ATTL3) |

A header-only neural network template library written in C++11. C-ATTL3 uses [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library. If GPU acceleration is enabled, it also utilizes NVIDIA's CUDA toolkit and specifically the [cuBLAS](https://developer.nvidia.com/cublas) library. C-ATTL3 allows for the easy construction and training of both feed-forward and recurrent neural networks ranging from simple MLPs and RNNs to state-of-the-art InceptionNets, ResNets, DenseNets,  and convolutional LSTMs. The library can handle data of different ranks and different floating point scalar types such as `float` and `double`. The Doxygen documentation of the library can be found [here](https://viktorc.github.io/C-ATTL3/html/annotated.html).

## Components
The following sub-sections describe the main components of the C-ATTL3 deep learning library. Knowledge of these components and their relations is required for the effective usage of the library.

### Layer
The lowest level building blocks of neural networks in C-ATTL3 are the layers. The library provides a wide selection of them that can be used for the construction of neural network modules. The available layer types are the following:
* [Layer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_layer.html) [A]
  * [KernelLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_kernel_layer.html) [A]
    * [DenseLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dense_layer.html)
    * [ConvolutionLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_convolution_layer.html) (3)
    * [DeconvolutionLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_deconvolution_layer.html) (3)
  * [ActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_activation_layer.html) [A]
    * [IdentityActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_identity_activation_layer.html)
    * [ScaledActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_scaled_activation_layer.html)
    * [BinaryStepActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_binary_step_activation_layer.html)
    * [SigmoidActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sigmoid_activation_layer.html)
    * [TanhActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_tanh_activation_layer.html)
    * [SoftplusActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_softplus_activation_layer.html)
    * [SoftmaxActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_softmax_activation_layer.html)
    * [ReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_re_l_u_activation_layer.html)
    * [LeakyReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_leaky_re_l_u_activation_layer.html)
    * [ELUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_e_l_u_activation_layer.html)
    * [PReLUActivationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_p_re_l_u_activation_layer.html)
  * [PoolLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_pooling_layer.html) [A] (3)
    * [SumPoolLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sum_pool_layer.html) (3)
    * [MaxPoolLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_max_pool_layer.html) (3)
    * [MeanPoolLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_mean_pool_layer.html) (3)
  * [BroadcastLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_broadcast_layer.html)
  * [BatchNormalizationLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_batch_normalization_layer.html)
  * [DropoutLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dropout_layer.html)
  * [ReshapeLayer](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_reshape_layer.html)

Most layers can handle data of rank 1 to 3 with the exception of convolutional and pooling layers which only accept rank 3 data. The actual rank of the input and output of the layers is one greater than the nominal rank of the layers to allow for batch learning. In the case of a layer with a nominal rank of 3, the input tensor is expected to be a rank-4 tensor with its ranks representing the sample number, height, width, and depth/channel (N,H,W,C) respectively. The nominal dimensionalities of the accepted input tensors of the different layers are specified using instances of the [Dimensions](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dimensions.html) class which relies on expression templates and compile time polymorphism to enable the fast and easy computation of the input dimensions of intermediary layers in complex neural networks. Besides the input dimensions, the one parameter required by all, most layers rely on multiple other hyper-parameters as well (e.g. max-norm constraint, dilation, receptor field size, etc.). These parameters may need to be fine-tuned manually or via random search (or in some other way) to optimize the behaviour of the networks.

#### WeightInitialization
The kernel layers (fully-connected and convolutional) also require weight initialization. The out-of-the-box weight initializations include:
* [WeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_weight_initialization.html) [A]
  * [ZeroWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_zero_weight_initialization.html)
  * [OneWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_one_weight_initialization.html)
  * [GaussianWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_gaussian_weight_initialization.html) [A]
    * [LeCunWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_le_cun_weight_initialization.html)
    * [GlorotWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_glorot_weight_initialization.html)
    * [HeWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_he_weight_initialization.html)
    * [OrthogonalWeightInitialization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_orthogonal_weight_initialization.html)

These weight initializations aim to mitigate the problem of vanishing and exploding gradients. Weight initialization can make or break neural networks, however the usage of batch normalization layers may reduce the networks' sensitivity to initialization.

#### ParameterRegularization
Parametric layers, i.e. layers with learnable parameters, also support parameter regularization. The standard regularization penalty functions are:
* [ParameterRegularization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_regularization_penalty.html) [A]
  * [NoParameterRegularization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_no_parameter_regularization.html)
  * [L1ParameterRegularization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l1_parameter_regularization.html)
  * [L2ParameterRegularization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l2_parameter_regularization.html)
  * [ElasticNetParameterRegularization](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_elastic_net_parameter_regularization.html)

These regularizations are counter measures against overfitting. They are especially useful in complex networks with a huge number of parameters. L1 regularization adds the sum of the absolute values of the parameters to the total loss, L2 regularization adds the sum of the squared values of the paramaters to the loss, and the elastic net regularization combines the other two.

### NeuralNetwork
The highest level building blocks of the different architectures are the neural network implementations provided by the library. Using these implementations, either as modules in a composite constellation or as standalone networks, almost any neural network architecture can be constructed. They are the following:
* [NeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_neural_network.html) [A]
  * [FeedforwardNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_feedforward_neural_network.html) (NS)
  * [UnidirectionalNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_unidirectional_neural_network.html) [A] (S)
    * [RecurrentNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_recurrent_neural_network.html) (S)
    * [LSTMNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_l_s_t_m_neural_network.html) (S)
  * [CompositeNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_composite_neural_network.html) [A]
    * [StackedNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_stacked_neural_network.html)
    * [ParallelNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_parallel_neural_network.html) (NS)
    * [ResidualNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_residual_neural_network.html) (NS)
    * [DenseNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_dense_neural_network.html) (NS)
    * [SequentialNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_sequential_neural_network.html) (S)
    * [BidirectionalNeuralNetwork](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_bidirectional_neural_network.html) (S)

These neural networks are either sequential (S) or non-sequential (NS). Non-sequential networks handle input tensors of rank 2 to 4 (one greater than their nominal ranks) where the first rank represents the samples thus allowing for (mini-) batch training. On the other hand, sequential networks handle inputs of rank 3 to 5 (two greater than their nominal ranks) where the first rank, similarly to that of non-sequential networks' inputs, represents the samples and the second rank represents the time steps. Generally, non-sequential data consists of independent data points, while sequential data is made up of sequences of dependent observations.

Feedforward neural networks are ordinary networks with a set of layers through which the non-sequential input data is propagated. Besides feedforward networks, the library also provides recurrent neural networks. Both vanilla recurrent neural networks and LSTMs support arbitrary output schedules and multiplicative integration; however, due to the fact that they are unrolled for propagation/backpropagation through time, their memory requirements can be fairly high (despite the shared parameters) depending on the number of time steps. In addition to the structurally simple networks mentioned above, C-ATTL3 offers a range of composite neural networks as well. The most straightforward example of such networks is the stacked neural net which is composed of one or more either sequential or non-sequential sub-nets that are appended to one another in a serial fashion. Such a stacked network of two sub-modules may be used for the implementation of an autoencoder for example. Parallel neural nets, similarly to stacked networks, contain one or more sub-modules. However, in the case of parallel networks, these modules are by definition non-sequential neural nets through which the input is simultaneously propagated and eventually merged via summation, multiplication, or concatenation along either the highest or the lowest (2nd) rank. Residual networks and dense networks are implementations of the [ResNet](https://arxiv.org/abs/1512.03385) and [DenseNet](https://arxiv.org/abs/1608.06993) architectures that use non-sequential neural networks as their sub-modules. Sequential networks function as wrappers around non-sequential networks allowing them to be used on sequential data by applying them to each time step. A bidirectional network takes a unidirectional network which it clones and reverses, yielding two recurrent networks processing the input data in opposite directions along the time-step rank. The outputs of the two unidirectional sub-nets of a bidirectional network can be joined by concatenation, summation, or multiplication.

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

These optimizers all aim to minimize a loss function with respect to the learnable parameters of the layers making up a neural network. The direction and magnitude of the updates depend on the update schemes of the optimizers and on the derivatives of the loss function with respect to the parameters as computed during back-propagation. Some of the optimizers maintain moving averages over the gradients to simulate momentum and reduce the chances of getting stuck in local minima, some use annealing to reduce the magnitude of updates over 'time' so that they can settle on a minimum instead of dancing around it, some use individual parameter-specific learning rates, and others use combinations of these techniques.

#### Loss
Similarly to the layers, optimizers rely on several hyper-parameters as well. Besides the hyper-parameters, optimizers also rely on 'practically' differentiable loss functions to minimize. The library provides the following out of the box loss functions:
* [Loss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_loss.html) [A]
  * [UniversalLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_universal_loss.html) [A]
    * [AbsoluteLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_absolute_loss.html)
    * [QuadraticLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_quadratic_loss.html)
    * [HingeLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_hinge_loss.html)
    * [CrossEntropyLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_cross_entropy_loss.html)
    * [SoftmaxCrossEntropyLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_softmax_cross_entropy_loss.html)
    * [MultiLabelHingeLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_multi_label_hinge_loss.html)
    * [MultiLabelLogLoss](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_multi_label_log_loss.html)

Given a loss function and some hyper-parameters, optimizers can be constructed and used for gradient checks (e.g. to test self-implemented sub-classes of the core interfaces) and network training. Both of these methods are parameterized by a neural network and data providers.

### DataProvider
Data providers are responsible for supplying the data used for gradient verification, training, and testing. The currently available data providers include:
* [DataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_data_provider.html) [A]
  * [PartitionDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_partition_data_provider.html)
  * [MemoryDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_memory_data_provider.html)
  * [JointFileDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_joint_file_data_provider.html) [A]
    * [CIFARDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_c_i_f_a_r_data_provider.html) (3,NS)
  * [SplitFileDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_split_file_data_provider.html) [A]
    * [MNISTDataProvider](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_m_n_i_s_t_data_provider.html) (3,NS)

The partition data provider maps to a continuous segment of the data backing another data provider. This allows for the partitioning of a single provider into training and test data providers. The memory data provider is backed by two in-memory tensors containing the observations and the objectives. The library also provides file-based data providers that allow for the training of networks without the need to load entire data sets into memory. The joint file data provider supports the processing of data sets stored in an arbitrary number of files that each contain both the observations and their respective objectives. On the other hand, the split file data provider is backed by an arbitrary number of file pairs for when the observations and the objectives are stored in separate files. C-ATTL3 includes a specialized data provider for the popular MNIST data set for the easy comparison of the performance of different network architectures to published results. The library also ships a specialized data provider for the CIFAR data set which supports both CIFAR-10 and CIFAR-100.

### Preprocessor
C-ATTL3 also contains a few preporcessors that can be used to transform the input data. They are:
* [Preprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_preprocessor.html) [A]
  * [NormalizationPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_normalization_preprocessor.html) (NS)
  * [PCAPreprocessor](https://viktorc.github.io/C-ATTL3/html/classcattle_1_1_p_c_a_preprocessor.html) (NS)

Both normalization preprocessors and PCA preprocessors support the centering and optionally the standardization of data. PCA preprocessors can also utilize whitening and, in case of single-channel data, dimensionality reduction.

## Usage
The following code snippets demonstrate the usage of the library via a simple example.
```cpp
using namespace cattle;
TensorPtr<double,4> training_obs_ptr(new Tensor<double,4>(80u, 32u, 32u, 3u));
TensorPtr<double,4> training_obj_ptr(new Tensor<double,4>(80u, 1u, 1u, 1u));
training_obs_ptr->setRandom();
training_obj_ptr->setRandom();
PCAPreprocessor<double,3> preproc;
preproc.fit(*training_obs_ptr);
preproc.transform(*training_obs_ptr);
MemoryDataProvider<double,3,false> training_prov(std::move(training_obs_ptr), std::move(training_obj_ptr));
```
To demonstrate the usage of the library's in-memory data provider, some random training data is generated. The training data is comprised of two tensors of rank 4 and type `double`; one for the observations and one for the objectives. The function to be approximated by the neural network is the mapping function between the observations and the objectives. The first rank of these tensors always denotes the samples and its value must be the same in the two tensors. In the example above, the training data consists of 80 observation-objective pairs. In case of sequential data, the second rank of the tensors denotes the time steps which can differ between the observations and the objectives (if the output sequence length of the network does not match the input sequence length); however, in this example, the tensors represent non-sequential data (see the third template argument of the data providers or the optimizer; or the fact that the network to be trained is an inherently non-sequential feed-forward neural network), thus the last 3 ranks describe the individual observation and objective instances. The nominal rank of the data here is thus 3; representing height, width, and depth. The observations are images with a resolution of 32x32 and 3 color channels, while the objectives are single scalars. The data is generated by filling the two tensors with random values between 0 and 1. A PCA preprocessor is then created and fit to the training observation set. After it is fit to the data, it is used to project it into the feature space where its variance along its features is the highest. Finally, the training data provider is created out of the two tensors by moving the two unique pointers referencing them to the `InMemoryDataProvider` constructor.
```cpp
TensorPtr<double,4> test_obs_ptr(new Tensor<double,4>(20u, 32u, 32u, 3u));
TensorPtr<double,4> test_obj_ptr(new Tensor<double,4>(20u, 1u, 1u, 1u));
test_obs_ptr->setRandom();
test_obj_ptr->setRandom();
preproc.transform(*test_obs_ptr);
MemoryDataProvider<double,3,false> test_prov(std::move(test_obs_ptr), std::move(test_obj_ptr));
```
The test data provider is created similarly. However, it is important not to re-fit the preprocessor to the observation data set to ensure that the same transformation is applied to both the training and the test data. This test data is used to assess the accuracy of the neural network on data it has not encountered during the training process. This provides a measure of the network's generalization ability; the difference between the network's accuracy on the training data and that on the test data is a metric of overfitting. The test data is usually a smaller portion of all the available data than the training data. In our example, it is 20 samples as opposed to the 80 comprising the training data. Note that all the other ranks of the test observation and objective tensors must match those of the training observation and objective tensors.
```cpp
auto init = std::make_shared<HeWeightInitialization<double>>();
auto reg = std::make_shared<L2ParameterRegularization<double>>();
std::vector<LayerPtr<double,3>> layers(9);
layers[0] = LayerPtr<double,3>(new ConvolutionLayer<double>(training_prov.get_obs_dims(), 10, init, reg, 5, 5, 2, 2));
layers[1] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[0]->get_output_dims()));
layers[2] = LayerPtr<double,3>(new MaxPoolLayer<double>(layers[1]->get_output_dims()));
layers[3] = LayerPtr<double,3>(new ConvolutionLayer<double>(layers[2]->get_output_dims(), 20, init, reg));
layers[4] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[3]->get_output_dims()));
layers[5] = LayerPtr<double,3>(new MaxPoolLayer<double>(layers[4]->get_output_dims()));
layers[6] = LayerPtr<double,3>(new DenseLayer<double,3>(layers[5]->get_output_dims(), 500, init, reg));
layers[7] = LayerPtr<double,3>(new ReLUActivationLayer<double,3>(layers[6]->get_output_dims()));
layers[8] = LayerPtr<double,3>(new DenseLayer<double,3>(layers[7]->get_output_dims(), 1, init, reg));
FeedforwardNeuralNetwork<double,3> nn(std::move(layers));
```
The next step is the construction of the neural network. The above snippet demonstrates that of a simple convolutional neural network. The neural network implementation used is `FeedforwardNeuralNetwork` which takes a vector of unique layer pointers. Each layer in the vector must have the same input dimensions as the output dimensions of the preceding layer. Notice how the dimensions of the outputs of the layers do not need to be calculated manually; they can be simply retrieved using the `get_output_dims` members of the previous layers. It should also be noted that all neural networks require their layers to be of the same nominal rank and scalar type as the network itself. The example network consists of convolutional, max pooling, rectified linear unit, and fully connected layers. Convolutional and fully connected layers require weight initialization; due to its well-known compatibility with ReLU activations, He weight initialization is a good choice in our situation. As the `WeightInitialization` class specifies a stateless interface, multiple layers can use the same implementation instance (this is the reason they take a shared pointer). The same can be said about the`ParameterRegularization` abstract type. All layers with learnable parameters, including the fully connected and convolutional ones above, support optional parameter regularization. In our example, the choice fell upon the popular L2 regularization penalty function for all parameteric layers. Similarly to the unique tensor pointer arguments of the data providers, the vector of unique layer pointers required by the network's constructor must be moved as well, as unique smart pointers cannot be copied.
```cpp
nn.init();
```
Once the network is constructed, it is appropriate to initialize it. An unitialized network is in an undefined state. The initialization of the network entails the initialization of all its layers' parameters. Care must be taken not to unintentionally overwrite learned parameters by re-initializing the network.
```cpp
auto loss = std::make_shared<QuadraticLoss<double,3,false>>();
NadamOptimizer<double,3,false> opt(loss, 20);
```
Having set up the data providers and the network, it is time to specify the loss function and the optimizer. For the sake of simplicity (concerning the data generation), the quadratic loss function is used in our example. Like `WeightInitialization`and `ParameterRegularization`, `Loss` also defines a stateless interface; this is why it is wrapped in a shared pointer and why a single instance can be used by multiple optimizers. The optimizer used in our example is the `NadamOptimizer` which is generally a good first choice. Note the consistency of the template arguments; the data providers, the preprocessor, the neural network, the loss function, and the optimizer must all have the same scalar type, rank, and sequentiality. As specified by the third argument of the optimizer's constructor, the batch size used for training and testing is 20. This means that both the training and the test data instances are processed in batches of 20. After the processing of each training batch, the parameters of the network's layers are updated. In our case, an epoch thus involves 4 parameter updates. It should be noted that most optimizers have several hyper-parameters that usually have reasonable default values and thus do not necessarily need to be specified.
```cpp
opt.optimize(nn, training_prov, test_prov, 500);
```	
With everything set up and ready, the optimization can commence. The four non-optional paramaters of the `optimize` method are the neural network whose paramaters are to be optimized, the training data provider, the test data provider, and the number of epochs for which the optimization should go on. For our optimizer, these 500 epochs mean 2000 parameter updates alltogether. The `optimize` method is moderately verbose; for every epoch, it prints the training and test losses to the standard out stream. It also prints a warning message in case the test loss is greater than at the previous epoch.
```cpp
Tensor<double,4> input(5u, 32u, 32u, 3u);
input.setRandom();
preproc.transform(input);
Tensor<double,4> prediction = nn.infer(input);
```	
The final code snippet demonstrates the usage of the trained neural network for inference. A random input tensor of the correct nominal input dimensions is generated, transformed using the PCA preprocessor, and fed to the `infer` method which has the neural network propagate the tensor through its layers and output its prediction. As seen above, inference is not restricted to single instances but can be performed on batches of data as well.

More examples of neural network constructs can be found [here](https://github.com/ViktorC/C-ATTL3/tree/master/examples).

## TODO
The list below contains the planned features of the library.
- [x] __Test automation using Google Test__
- [ ] CMake build
- [x] Deconvolutional layer
- [x] Broadcast and reshaping layers
- [ ] Temporal (convolutional) networks
- [ ] Convolution, deconvolution and pooling for 1st and 2nd degree features
- [ ] More comprehensive GPU support and cuDNN utilization
- [ ] Data pre-fetching for file based providers
- [x] A codec for at least one or two simple image formats such as PPM
- [ ] __Network serialization and de-serialization__
- [ ] AMSGrad optimizer
- [ ] Hessian-free optimizer with conjugate gradient descent
- [ ] L-BFGS optimizer
- [ ] Particle swarm optimizer
- [ ] GA optimizer
- [ ] PBIL optimizer
- [ ] FFT and/or Winograd filtering for CPU convolution
- [ ] LDA preprocessor
- [ ] GRU network
- [ ] CTC loss
