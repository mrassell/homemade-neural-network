# Handwritten Digit Classification Neural Network

**This project implements a neural network from scratch (No TensorFlow or Pytorch) using NumPy to classify handwritten digits from the MNIST dataset.**

Credit goes to Samson Zhang's YouTube Video "Building a neural network FROM SCRATCH" which taught me the linear algebra. I included my notes in a Notion doc below
[https://ordinary-health-cab.notion.site/Image-Classification-Neural-Network-FROM-SCRATCH-72aa4de370f54d5d8a30dd50da0a92f3](url)

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
- The training set: teaches the model
- The development set: evaluate model's performace on unseen data (Separate from teaching, detects overfitting, 
  or if the model performs well on training data but poorly on dev data)
- Input layer: 784 nodes (28x28 pixel images) 
- Hidden layer: 10 nodes with ReLU activation, without this we would just be calculating linear combinations and we want more than just a fancy linear regression 
- Output layer: 10 nodes with Softmax activation, converting output layer into probabilities
  
Method: **Forward propagation** will apply weights and biases to make predictions, but we need to run an algorithm to optimize weights and biases with **back propagation**. 

Essentially starting with our prediction and finding out by how much it deviates from our actual label, we see how much each of those weights and biases contributed to that error.
We take our predictions and subtract the actual label and one-hot encode the label so that it is in a binary vector format that is compatible with neural network outputs because they have multiple classes and we do not want to create false ordinal relationships between classes.

Using derivative of loss function with respect to weights and using derivative of activation function so we can see how much we should nudge the weights and biases in the layers. 

Then we update parameters accordingly with some learning rate that we set, alpha (hyperparameter).

Gradient descent: Once we update parameters, we iterate through the entire propagation and adjusting process again.

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
