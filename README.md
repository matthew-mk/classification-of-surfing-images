# Classification of Surfing Images

An object-oriented system that can be used to create, train, test, and evaluate machine learning models that classify images of surfing locations based on the quality of surfing conditions. Convolutional Neural Network (CNN) models can be created using Keras and a variety of models can be created using Scikit-learn, including Support Vector Machine (SVM), Random Forest (RF), and K-nearest Neighbors (KNN) models.

## Project Structure

```
├── datasets
│   ├── binary                              - Datasets for binary classification. Images are classed as surfable or unsurfable.
│   └── rating                              - Datasets for multiclass classification. Images are rated from 1-5.
│
├── documentation                           - Project documentation.
├── saved_models 	                    - The models that are saved by the user.
└── src    		   		    - The source code.
    ├── classes
    │   ├── cnn.py                          - Classes for creating CNN models using Keras.
    │   ├── dataset_handler.py              - Classes for loading and preprocessing datasets.
    │   └── sklearn.py                      - Classes for creating models using Scikit-learn.
    │
    ├── mains	      	                    - The 'main' files where models are initialized, trained, tested, and evaluated.
    ├── tests	      	                    - Functionality for testing the code.
    └── utils
        ├── helper_utils.py                 - Utility functions that help with various parts of the system.
        └── train_test_utils.py             - Utility functions for training, testing, and evaluating models.
```

## Installation & Setup

-

## How To Run The Code

-

## Acknowledgements

- I would like to thank my supervisor, Dr. John Wilson, for providing invaluable advice and guidance throughout the course of the project.
