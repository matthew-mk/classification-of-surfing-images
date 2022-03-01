# Classification of Surfing Images

An object-oriented system that can be used to create, train, test, and evaluate machine learning models that classify images of surfing locations based on the quality of surfing conditions. Convolutional Neural Network (CNN) models can be created using Keras and a variety of models can be created using Scikit-learn, including Support Vector Machine (SVM), Random Forest (RF), and K-nearest Neighbors (KNN) models.

## Installation

-

## Project Structure

```
├── datasets                                - Datasets containing images from different surfing locations.
├── documentation                           - Project documentation.
├── saved_models 	                        - Stores the models that the user chooses to save.
└── src    		   		                    - The source code.
    ├── classes
    │   ├── cnn.py                          - Classes for creating CNN models using Keras.
    │   ├── dataset_handler.py              - Classes for loading and preprocessing datasets for CNN and Scikit-learn models.
    │   └── sklearn.py                      - Classes for creating models using Scikit-learn.
    │
    ├── mains	      	                    - Main files where models are initialized, trained, tested, and evaluated.
    ├── tests	      	                    - Functionality for testing the code.
    └── utils
        ├── helper_utils.py                 - Utility functions that help with various areas of the system.
        └── train_test_utils.py             - Utility functions for training, testing, and evaluating models.
```

## Maintenance

-

## Acknowledgements

I would like to thank my supervisor, Dr. John Wilson, for providing invaluable advice and guidance throughout the course of the project.
