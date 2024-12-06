[![ML Pipeline](https://github.com/SunilKV123/ERAv3-Session6/actions/workflows/ml_pipeline.yml/badge.svg)](https://github.com/SunilKV123/ERAv3-Session6/actions/workflows/ml_pipeline.yml)

# MNIST CNN Model with CI/CD Pipeline

This project implements a CNN model for MNIST digit classification with a complete CI/CD pipeline.

    mnist_pipeline/
    │
    ├── .github/
    │ └── workflows/
    │ └── ml_pipeline.yml # GitHub Actions workflow configuration
    │
    ├── src/
    │ ├── init.py # Makes src a Python package
    │ ├── model.py # CNN model architecture
    │ ├── train.py # Training script
    │ └── utils.py # Utility functions
    │
    ├── tests/
    │ ├── init.py # Makes tests a Python package
    │ ├── test_model.py # Model architecture tests
    │ └── test_training.py # Training pipeline tests
    │
    ├── models/ # Directory for saved models
    │ └── .gitkeep # Keeps empty models directory in git
    │
    ├── .gitignore # Specifies which files Git should ignore
    ├── requirements.txt # Project dependencies for GPU development
    ├── requirements-cpu.txt # Project dependencies for CPU-only execution
    ├── setup.py # Package installation configuration
    └── README.md # Project documentation

## Requirements
- Python 3.8+
- PyTorch (CPU or GPU version)
- torchvision
- pytest

## Installation

### For GPU Development
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```
### For CPU-Only Development or CI/CD
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
pip install -r requirements-cpu.txt
pip install -e .
```
## Local Testing
```bash
python -m unittest discover tests/
python src/train.py
```

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up a Python environment
2. Installs CPU-only dependencies
3. Runs all tests
4. Trains the model
5. Saves the trained model as an artifact

## Model Details
- Architecture: Convolutional Neural Network (CNN)
- Parameters: <20,000
- Input shape: 28x28 grayscale images
- Output: 10 classes (digits 0-9)
- Target accuracy: >99.4% Validation Accuracy

## Model Artifacts
Models are automatically saved with the following naming convention:
`mnist_model_<accuracy>acc_<timestamp>.pth`

Training log from the Google Colab:

    Using device: cuda
    Total trainable parameters: 19704
    Epoch 1/20 [Train]: 100%|██████████| 391/391 [00:39<00:00,  9.90it/s, loss=0.4761, acc=86.49%]
    Epoch 1/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.03it/s, loss=0.0734, acc=97.79%]

    Epoch 1/20:
    Train Loss: 0.4761, Train Accuracy: 86.49%
    Val Loss: 0.0734, Val Accuracy: 97.79%

    Epoch 2/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.03it/s, loss=0.1131, acc=96.83%]
    Epoch 2/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.91it/s, loss=0.0597, acc=98.25%]

    Epoch 2/20:
    Train Loss: 0.1131, Train Accuracy: 96.83%
    Val Loss: 0.0597, Val Accuracy: 98.25%

    Epoch 3/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.07it/s, loss=0.0896, acc=97.31%]
    Epoch 3/20 [Val]: 100%|██████████| 79/79 [00:03<00:00, 25.21it/s, loss=0.0477, acc=98.65%]

    Epoch 3/20:
    Train Loss: 0.0896, Train Accuracy: 97.31%
    Val Loss: 0.0477, Val Accuracy: 98.65%

    Epoch 4/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.10it/s, loss=0.0764, acc=97.74%]
    Epoch 4/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 31.91it/s, loss=0.0391, acc=98.94%]

    Epoch 4/20:
    Train Loss: 0.0764, Train Accuracy: 97.74%
    Val Loss: 0.0391, Val Accuracy: 98.94%

    Epoch 5/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.05it/s, loss=0.0673, acc=97.95%]
    Epoch 5/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.30it/s, loss=0.0327, acc=98.99%]

    Epoch 5/20:
    Train Loss: 0.0673, Train Accuracy: 97.95%
    Val Loss: 0.0327, Val Accuracy: 98.99%

    Epoch 6/20 [Train]: 100%|██████████| 391/391 [00:39<00:00,  9.91it/s, loss=0.0620, acc=98.15%]
    Epoch 6/20 [Val]: 100%|██████████| 79/79 [00:03<00:00, 26.27it/s, loss=0.0324, acc=99.05%]

    Epoch 6/20:
    Train Loss: 0.0620, Train Accuracy: 98.15%
    Val Loss: 0.0324, Val Accuracy: 99.05%

    Epoch 7/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.03it/s, loss=0.0582, acc=98.24%]
    Epoch 7/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.17it/s, loss=0.0295, acc=99.10%]

    Epoch 7/20:
    Train Loss: 0.0582, Train Accuracy: 98.24%
    Val Loss: 0.0295, Val Accuracy: 99.10%

    Epoch 8/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.10it/s, loss=0.0534, acc=98.38%]
    Epoch 8/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.35it/s, loss=0.0305, acc=99.13%]

    Epoch 8/20:
    Train Loss: 0.0534, Train Accuracy: 98.38%
    Val Loss: 0.0305, Val Accuracy: 99.13%

    Epoch 9/20 [Train]: 100%|██████████| 391/391 [00:39<00:00, 10.02it/s, loss=0.0531, acc=98.36%]
    Epoch 9/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 28.15it/s, loss=0.0302, acc=99.04%]

    Epoch 9/20:
    Train Loss: 0.0531, Train Accuracy: 98.36%
    Val Loss: 0.0302, Val Accuracy: 99.04%

    Epoch 10/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.12it/s, loss=0.0500, acc=98.46%]
    Epoch 10/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 31.68it/s, loss=0.0273, acc=99.20%]

    Epoch 10/20:
    Train Loss: 0.0500, Train Accuracy: 98.46%
    Val Loss: 0.0273, Val Accuracy: 99.20%

    Epoch 11/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.13it/s, loss=0.0482, acc=98.56%]
    Epoch 11/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.02it/s, loss=0.0261, acc=99.19%]

    Epoch 11/20:
    Train Loss: 0.0482, Train Accuracy: 98.56%
    Val Loss: 0.0261, Val Accuracy: 99.19%

    Epoch 12/20 [Train]: 100%|██████████| 391/391 [00:39<00:00,  9.99it/s, loss=0.0474, acc=98.65%]
    Epoch 12/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 28.88it/s, loss=0.0246, acc=99.26%]

    Epoch 12/20:
    Train Loss: 0.0474, Train Accuracy: 98.65%
    Val Loss: 0.0246, Val Accuracy: 99.26%

    Epoch 13/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.05it/s, loss=0.0463, acc=98.61%]
    Epoch 13/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.07it/s, loss=0.0257, acc=99.26%]

    Epoch 13/20:
    Train Loss: 0.0463, Train Accuracy: 98.61%
    Val Loss: 0.0257, Val Accuracy: 99.26%

    Epoch 14/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.07it/s, loss=0.0435, acc=98.72%]
    Epoch 14/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.04it/s, loss=0.0250, acc=99.34%]

    Epoch 14/20:
    Train Loss: 0.0435, Train Accuracy: 98.72%
    Val Loss: 0.0250, Val Accuracy: 99.34%

    Epoch 15/20 [Train]: 100%|██████████| 391/391 [00:39<00:00,  9.96it/s, loss=0.0438, acc=98.65%]
    Epoch 15/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.52it/s, loss=0.0233, acc=99.27%]

    Epoch 15/20:
    Train Loss: 0.0438, Train Accuracy: 98.65%
    Val Loss: 0.0233, Val Accuracy: 99.27%

    Epoch 16/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.11it/s, loss=0.0406, acc=98.77%]
    Epoch 16/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.39it/s, loss=0.0216, acc=99.41%]

    Epoch 16/20:
    Train Loss: 0.0406, Train Accuracy: 98.77%
    Val Loss: 0.0216, Val Accuracy: 99.41%

    Epoch 17/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.12it/s, loss=0.0424, acc=98.70%]
    Epoch 17/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 31.07it/s, loss=0.0228, acc=99.38%]

    Epoch 17/20:
    Train Loss: 0.0424, Train Accuracy: 98.70%
    Val Loss: 0.0228, Val Accuracy: 99.38%

    Epoch 18/20 [Train]: 100%|██████████| 391/391 [00:39<00:00, 10.02it/s, loss=0.0395, acc=98.78%]
    Epoch 18/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 31.67it/s, loss=0.0226, acc=99.28%]

    Epoch 18/20:
    Train Loss: 0.0395, Train Accuracy: 98.78%
    Val Loss: 0.0226, Val Accuracy: 99.28%

    Epoch 19/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.12it/s, loss=0.0416, acc=98.72%]
    Epoch 19/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 32.22it/s, loss=0.0241, acc=99.30%]

    Epoch 19/20:
    Train Loss: 0.0416, Train Accuracy: 98.72%
    Val Loss: 0.0241, Val Accuracy: 99.30%

    Epoch 20/20 [Train]: 100%|██████████| 391/391 [00:38<00:00, 10.12it/s, loss=0.0376, acc=98.84%]
    Epoch 20/20 [Val]: 100%|██████████| 79/79 [00:02<00:00, 31.73it/s, loss=0.0204, acc=99.41%]
    Epoch 20/20:
    Train Loss: 0.0376, Train Accuracy: 98.84%
    Val Loss: 0.0204, Val Accuracy: 99.41%

    Training completed with validation accuracy: 99.41%
    Model saved as: mnist_model_99.41acc_20241201_131309.pth




## GitHub Actions
The pipeline runs automatically on every push to the repository:
- Uses CPU-only PyTorch version
- Executes all tests
- Trains model
- Stores model artifacts
- Results viewable in Actions tab

## Development Notes
- Local development supports both CPU and GPU
- CI/CD pipeline runs on CPU only
- All tests must pass for successful deployment
- Model artifacts are retained for 90 days


## Notes
- The model is trained on CPU in GitHub Actions
- Trained models are saved as artifacts in the workflow
- All tests must pass for successful deployment


## To use this project:
1. Clone the repository
2. Create a new branch for your changes
3. Push your changes to the new branch
4. Create a pull request to merge your changes into the main branch
5. The workflow will automatically run and deploy your changes if all tests pass
