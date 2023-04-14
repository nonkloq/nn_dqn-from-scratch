# Neural Network and Deep Q-Learning Network from Scratch

Implementation of a Neural Network (NN) and Deep Q-Learning Network (DQN) using only the numpy library and some simple math. The NN includes Stochastic Gradient Descent (SGD) for weight updating, and a DQN is trained to play the game of Cartpole.

## Multilayered Perceptron (NN) Using SGD Algorithm

### Neural Network Construction: [Notebook](nn-mlp_from_scratch.ipynb)

This notebook provides a step-by-step procedure for constructing a multilayered perceptron.

### Neural Network Implementation: [NeuralNetwork](nn.py)

This file contains the full implementation of the neural network, with added momentum to the weight updating step. To save and load the `NeuralNetwork`, use `save_network` and `load_network` from [saveload.py](saveload.py).

## Deep Q-Learning Network

### Train DQN to Play Cartpole: [Notebook](dqn_from_scratch.ipynb)

This notebook demonstrates how to use the `NeuralNetwork` to implement the DQN algorithm.

### Custom Gym Environment

**Maze Harvest:** [Environment](maze_harvest.py)
> Check the Agent Training Notebook to learn more about the environment.

### DQN Using TensorFlow to Play Maze Harvest

**DQN Using TensorFlow:** [DQN](dqn_tf.py)

**Agent Training:** [Notebook](maze_harvest_train_tf.ipynb)

### Networks Folder

This folder contains pre-trained networks. Refer to the notebooks to learn how to load and use the networks.

## License

This project is licensed under the terms of the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
