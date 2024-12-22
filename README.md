## Project Overview

This project is a **trash classification model** that uses a simple **Convolutional Neural Network (CNN)** for classifying images of trash into six categories:
1. Cardboard
2. Glass
3. Metal
4. Paper
5. Plastic
6. Trash

The model is trained using the **TrashNet** dataset, which contains labeled images of various types of trash. This project includes code for downloading and preprocessing the dataset, training the model, and making predictions.

## Requirements

Before running the code, make sure you have the following libraries installed:

- Python 3.x
- PyTorch
- Torchvision
- Pillow
- Matplotlib
- Datasets
- Wandb

You can install all the required dependencies using the requirements.txt file:
```bash
pip install -r requirements.txt
```

## Dataset
The model uses the TrashNet dataset available on the Hugging Face hub.

To load the dataset, the following code is used:

from datasets import load_dataset
```bash
ds = load_dataset("garythung/trashnet")
The dataset consists of images categorized into six classes, and the images are saved locally for training the model.
```

## Setup Intructions
1. Clone repository
``` bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```

2. Install dependence
If you don't have the required packages already installed, run:
```bash
pip install -r requirements.txt
```

3. Download and preprocess the dataset:
The dataset is automatically downloaded and preprocessed during the first run. It will be stored in the ./data_trash directory.

4. Set up Wandb (optional):
If you want to track your experiments with Weights & Biases (Wandb), ensure that you have a Wandb account and login using:
```bash
wandb.login()
```

## Training the model
Once the environment is set up, you can train the model by running the following script:
```bash
python trash-classification.py
```

## Prediction
Once the model is trained, you can use it to predict new images. Use the following function to predict a single image:
```bash
predict_image('path_to_your_image.jpg', model, device)
```

## GitHub Actions Workflow
This repository includes a GitHub Actions workflow for automating the model training process. The workflow is located in the .github/workflows directory and runs on every push to the repository. It will:

- Set up the environment.
- Install dependencies from requirements.txt.
- Train the model.
- Log training results to Wandb.

Once the environment is set up, you can train the model by running the following script:
python trash-classification.py
