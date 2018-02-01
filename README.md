# C-B3Y0ND
A neural network library written in C++. C-B3Y0ND uses [Eigen](http://eigen.tuxfamily.org), the popular linear algebra library when run on the CPU. It allows for the easy construction and training of feed-forward neural networks ranging from simple MLPs to state-of-the-art convolutional InceptionNets, ResNets, Inception-ResNets, and DenseNets. It contains a wide selection of layers that can be used for the construction of neural networks. The available layer types are the following:
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

Besides the input dimensions, the one parameter required by all, each layer uses multiple hyper-parameters (e.g. max-norm constraint or dilation). These parameters can be fine-tuned to optimize the behaviour of the networks. The fully-connected and convolutional layers also require weight initializers. The out-of-the-box weight initializers include:
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

Similarly to the layers, these optimizers rely on hyper-parameters as well. Besides the hyper-parameters, optimizers also require a more-or-less differentiable loss function and regularization penalty function. The library provides the following out of the box loss functions:
