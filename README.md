# Digit Recognizer

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Project Description

This project implements a convolutional neural network (CNN) for recognizing handwritten digits. Using the MNIST dataset, I've trained a model that achieves 97% accuracy on Kaggle's test set.

## Data Source

The data for this project comes from the [Digit Recognizer competition on Kaggle](https://www.kaggle.com/c/digit-recognizer). It's a classic computer vision dataset that contains tens of thousands of handwritten images. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- Make (for using Makefile commands)

### Installation

1. Clone the repository:

```
git clone https://github.com/dtiourine/digit-recognizer.git
cd digit-recognizer
```

2. Install the required packages:

```
pip install -r requirements.txt
```

### Downloading the Data

To download the dataset from Kaggle:

1. Ensure you have a Kaggle account and have accepted the competition rules.
2. Place your Kaggle API token in `~/.kaggle/kaggle.json`.
3. Run:

```
make data
```

### Training the Model

To train the model:

```
make train
```

### Making Predictions

To make predictions on the test set:

```
make predict
```

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for digit_recognizer
│                         and configuration for tools like black
│
│
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── digit_recognizer                <- Source code for use in this project.
    │
    ├── __init__.py    <- Makes digit_recognizer a Python module
    │
    ├── dataset.py           <- Scripts to download or generate data
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualize.py  <- Scripts to visualize the data
```

## Results

The model achieved 97% accuracy on the Kaggle test set, demonstrating strong performance in digit recognition.

## Future Work

- Experiment with different CNN architectures
- Implement data augmentation techniques
- Try transfer learning with pre-trained models

## Acknowledgments

- Kaggle for providing the dataset
- The Cookiecutter Data Science project template

--------
