This repository contains a machine learning model for classifying trash images into categories such as cardboard, glass, metal, paper, plastic, and trash. The model is built using **PyTorch**, and the dataset is the **TrashNet** dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Setup Instructions](#setup-instructions)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
- [GitHub Actions Workflow](#github-actions-workflow)
- [Acknowledgments](#acknowledgments)

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






