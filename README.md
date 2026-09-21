# Neural Network from Scratch: MNIST Classifier

A multi-layer perceptron (MLP) built and trained from scratch in PyTorch to classify handwritten digits (MNIST), as part of building foundational deep learning intuition from first principles.

## What this does
- Loads the MNIST dataset (60,000 training / 10,000 test images)
- Defines a simple feedforward neural network: 784 → 128 (ReLU) → 10
- Trains using cross-entropy loss and the Adam optimizer
- Achieves ~97% test accuracy after 3 epochs
- Visualizes misclassified examples to understand model failure modes

## Why
Built as part of a self-directed deep dive into AI/ML fundamentals including understanding the mechanics of training (forward pass, backpropagation, gradient descent) rather than just using high-level libraries.

## Results
- Test accuracy: 96.9%
- See `wrong_predictions.png` for example misclassifications. Many are genuinely ambiguous even to a human eye (e.g. a "5" with a closed loop resembling a "6")

## Stack
Python, PyTorch, torchvision, matplotlib

## Next steps
Building toward multi-head self-attention and transformer architecture from scratch.
