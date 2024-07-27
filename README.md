# Handwritten Digit Classification Neural Network

**This project implements a neural network from scratch (No TensorFlow or Pytorch) using NumPy to classify handwritten digits from the MNIST dataset.**

Credit goes to Samson Zhang's YouTube Video "Building a neural network FROM SCRATCH" which taught me the linear algebra 
(I included my notes in a Notion doc below)

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.x
- pip (Python package installer)

## Setup

1. Clone the repository:
``` 
git clone https://github.com/mrassell/handwritten-digit-classification.git
cd handwritten-digit-classification
```
2. Install the required libraries:
```
pip install numpy pandas matplotlib
``` 
3. Download the MNIST dataset:
- Download the `train.csv` file from [Kaggle's MNIST dataset](https://www.kaggle.com/competitions/digit-recognizer/data)
- Place the `train.csv` file in the project directory

4. Update the `dataPath` variable in the script:
- Open `digit_classification.py`
- Modify the `dataPath` variable to point to your `train.csv` file location

## Running the Script

Execute the script:
```
python digit_classification.py
```
The script will:
1. Load and preprocess the MNIST data
2. Initialize the neural network parameters
3. Train the model using gradient descent
4. Display the training accuracy every 10 iterations
5. Test the model on a few sample images

## Understanding the Code

The neural network consists of:
- Input layer: 784 nodes (28x28 pixel images)
- Hidden layer: 10 nodes with ReLU activation
- Output layer: 10 nodes with Softmax activation

Key functions:
- `init_params()`: Initialize weights and biases
- `forward_prop()`: Perform forward propagation
- `backward_prop()`: Perform backward propagation
- `gradient_descent()`: Train the model
- `make_predictions()`: Use the trained model to make predictions
- `test_prediction()`: Visualize predictions on sample images

## Customization

You can modify the following parameters in the `gradient_descent()` function call:
- Learning rate (currently 0.10)
- Number of iterations (currently 500)

Also feel free to change the first parameter of the test predictions at the bottom to test different images of handwritten digits
```
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(230, W1, b1, W2, b2)

```

## Results

The script will display the model's accuracy during training and show predictions for a few sample images.
The average accuracy I achieved through multiple trials of 500 iterations was 85%.

## Note

This implementation is for educational purposes and may not be optimized for large-scale use. For production environments, consider using established machine learning libraries like TensorFlow or PyTorch.
