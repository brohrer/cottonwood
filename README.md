# The Cottonwood Machine Learning Framework

Cottonwood is built to as flexible as possible, top to bottom.
It's designed to minimize the iteration time when running experiments
and testing ideas. It's meant to be tweaked. Fork it. Add to it. Customize it
to solve the problem at hand.
[Why another framework?](https://end-to-end-machine-learning.teachable.com/blog/171633/cottonwood-flexible-neural-network-framework)

This code is always evolving. I recommend referencing a specific tag
whenever you use it in a project. Tags are labeled v1, v2, etc. and
the code attached to each one won't change.

If you want to follow along with the construction process for Cottonwood,
you can get a step-by-step walkthrough in End-to-End Machine Learning
[Course 312](https://end-to-end-machine-learning.teachable.com/p/write-a-neural-network-framework/)
and
[Course 313](https://end-to-end-machine-learning.teachable.com/p/advanced-neural-network-methods/)

## Try it out

```bash
git clone --branch v5 https://github.com/brohrer/cottonwood.git
python3 cottonwood/demo.py
```

## Revision history

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
