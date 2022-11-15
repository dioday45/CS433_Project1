# Higgs boson machine learning challenge - CS433

This is the repository for Project 1 of course CS433 at EPFL.

In this project, we present our methodology to answer the Higgs boson machine learning challenge which consists in building a binary classifier to predict whether an event corresponds to the decay of a Higgs boson or not. In other words, we will train machine learning model on precollected data to further predict wether a collision was **signal** or **background**.

### Models

To assess this challenge, we built multiple machine learning models. Theses models are the following:

- Linear Regression using gradient descent and stochastic gradient descent
- Least squares regression using normal equations
- Ridge Regression
- Logistic Regression using gradient descent
- Regularized Logistic Regression using gradient descent
- Neural Network

Without surprise, our best model is the Neural Network with an accuracy of 0.839 on the testing set. Our submission to the competition on AIcrowd is accessible through this [link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/submissions/203785) (our team name is CoucouRobinou).

## Structure

This repository is structured as described below:

- Data: Contains training and testing data (as .csv)
- Implementations.py: Requested models
- Models: This folder contains our pretrained weights used in the run.py script
- Notebooks:
  - EDA: Exploratory Data Analysis of the dataset
  - NN_training: Training process of our Neural Network
  - Data_processing: Preprocessing of the training dataset
  - Grid_search: Evaluation of the performances of our models
- run.py: Python script using pretrained weights to predict label on the testing set
- src
  - helpers.py: Helper functions used in our project
  - nn.py: Our Neural Network

```pseudocode
├── data
│   ├── train.csv
│   └── test.csv
├── implementations.py
├── models
├── notebooks
│   ├── EDA.ipynb
│   ├── NN_training.ipynb
│   ├── data_processing.ipynb
│   └── grid_search.ipynb
├── run.py
└── src
    ├── helpers.py
    └── nn.py
```

## Run prediction

**Important**: Due to their size, test.csv and train.csv are not included in this repository. To use our script, you must download them from this [link](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs), create a ``data/``folder and and put them in it (see above for correct structure).

To launch the prediction on the testing set, you can run the following command: 

```bash
python3 run.py
```

The script will first load the testing set (test.csv) and preprocess it. It will then make the prediction (either 1 or -1) and save them into a new .csv file.

### Used library

For this project, we used the following python library (make sure these libraries are correctly installed):

- Numpy
- Matplotlib
- csv

## Authors

- Daniel Tavares
- Thomas Castiglione
- Jeremy Di Dio
