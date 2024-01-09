# NN: Bare-Bones Neural Network in Numpy
A minimalist implementation of a neural network in Python, using Numpy. This project is designed to showcase the fundamentals of neural network architecture and operations, specifically tailored for classification tasks.

## Features
Linear Layer with Advanced Optimization: Incorporates the Adam optimization algorithm, as detailed in Adam: A Method for Stochastic Optimization. This allows for efficient and effective training iterations.
Xavier Glorot Initialization: Utilizes the Xavier Glorot method for initializing neural network weights, optimizing the learning process. For more details, see Understanding the difficulty of training deep feedforward neural networks.
* Modular Design: The project is structured into three main classes:
  * Layer: Handles the computations for a single linear layer, including the Adam update method.
  * NN: Manages a sequence of layers, capable of performing forward propagation, backpropagation, and accuracy evaluation.
* Dataset: Designed for handling a numpy dataset, it manages test-train splitting, shuffling, and batching.

## Requirements
Numpy: For all numerical computations.
SciPy: Optional, but recommended for additional scientific computing features.
Getting Started
To get started with NN, clone this repository and ensure you have the necessary dependencies installed:
```
git clone https://github.com/[your-github-username]/NN.git
cd NN
pip install numpy scipy
```

## How to Use NN
NN is designed to be straightforward to use, especially for training on the MNIST dataset. Below are the basic steps to get started:

### Quick Start
If you want to dive right in, you can simply run the main script for training on the MNIST dataset. The main script contains an example of how to use the network.

### Step-by-Step Guide
For a more detailed understanding, here's a breakdown of the key steps:

1. **Import the Classes:**
´´´
from NN import NN, Dataset
´´´

2. **Prepare Your Dataset:**
Create a Dataset instance with your data and labels. Here's an example with a batch size of 32:
´´´
dataset = Dataset(my_data_np, my_labels_np, batch_size=32)
´´´

3. **Initialize the Network:**
Create an NN instance with the desired layer sizes:
´´´
network = NN(\[size_of_each_layer\])
´´´

4. **Training:**
* Use ´(x,t)=dataset.batch_generator()´ to shuffle the data and yield new batches.
* For each batch, run ´network.forward(x)´ on the input data x.
* Perform backpropagation with ´network.backward(y-t)´, where y is the network output and t is the target.

### Understanding the Code
The if __name__ == '__main__': block in the script provides a practical example of how to train the network. For more detailed functionality, you can look at the code to understand the specific methods and their purposes.


## Contributing
Contributions to NN are welcome! Whether it's extending the functionality, improving the efficiency, or refining the documentation, your help is appreciated.

Usage and Attribution
This project is a bare-bones implementation of a neural network in Numpy, and I'm happy to share it with anyone interested in learning or building upon this work. If you use or adapt this code, I only ask that you give appropriate credit. Here's a suggested format for citation:

"Based on NN by Mikkel Opperud, available at github.com/Fjallar/NN/"

No formal license is applied to this project, and it's available for educational, research, and commercial use.
