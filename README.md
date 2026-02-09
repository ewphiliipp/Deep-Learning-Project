# Deep-Learning-Project

# Start Here
> **Start Here:** [Click here to open the Main Project Notebook](./Project.ipynb)

### Setup & Installation
Environment: Python 3.8+, PyTorch (CUDA-compatible).

Dependencies:
pip install torch torchvision numpy matplotlib seaborn pandas scikit-learn osculari tensorboard

Data Requirements:
CIFAR-10 / MNIST: Auto-download via torchvision.datasets.
CIFAR-10H Labels: cifar10h-probs.npy must be placed in the root directory.

### Execution Guide

Session Initialization: tmux new -s deep_learning_project
Training: Execute PhilNet/VAE training blocks.
Monitoring: tensorboard --logdir runs/ --port 6006
Hardware: Minimum 5GB free storage and CUDA-enabled GPU recommended.
