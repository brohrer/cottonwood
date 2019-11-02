## Cottonwood revision history

### v7

Reporting added 
as detailed in section 6 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* `__str__()` methods added to all core classes
* String representations of components combined together in a model report
* Widespread refactoring 
* Added to PyPi

### v6

Initializers added
as detailed in section 5 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* Glorot initialization
* He initialization
* Uniform initialization

### v5

Optimizers added
as detailed in section 4 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* Stochastic Gradient Descent (SGD)
* SGD with momentum
* Adam
* Noisy momentum (experimental)

#### Refactoring
* Creation of `data`, `examples`, and `experimental` directories.
* Variable, class, and module name changes throughout to better reflect
their purpose.
* Regularizers' function broken out into pre- and post-optimizer.

### v4

Custom layers added
as detailed in section 3 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* Normalization layer
* Difference layer
* Directed acyclic graph architectures
* Input value range inferred automatically

#### Bugfixes
* Error functions take single array as input, return single value.
* Connections in autoencoder visualization reflect connection weights

### v3

Dropout added
as detailed in section 2 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* Dropout

### v2

Regularization added, as detailed in section 1 of
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/).

#### Features
* **Regularizers** L1, L2, and Limit (max-norm)

### v1

The initial state of the code, as built in
[Course 312](https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework/)

#### Features
* Fully-connected layers
* Sequential architecture
* Backpropagation
* **Activation functions** hyperbolic tangent, rectified linear unit, and logistic
* **Error functions** squared error and absolute error

#### Examples
* Autoencoder

#### Data sets
* 2x2 pixel test images
* 3x3 pixel test images
* 7x7 pixel Nordic rune images

