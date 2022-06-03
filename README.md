# Classification of Surfing Images

This software was developed as a fourth-year dissertation project at the University of Strathclyde.

## Abstract

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
    ├── results	      	                    - Charts that display the results gathered during the evaluation process.
    ├── tests	      	                    - Unit tests.
    └── utils
        ├── helper_utils.py                 - Utility functions that help with various parts of the system.
        └── train_test_utils.py             - Utility functions for training, testing, and evaluating models.
```

## Installation & Setup

The instructions assume that PyCharm is used as the IDE. If you do not currently have PyCharm installed, the free community edition can be downloaded from https://www.jetbrains.com/pycharm/download/. Anaconda is also required, which is available to download at: https://www.anaconda.com/.

1. Open the project in PyCharm.
2. Click on &lt;No interpreter&gt; in the bottom right hand corner and select ‘Add environment’.
3. Select ‘Conda Environment’.
4. Select ‘New environment’ and use Python version 3.8.
5. Click OK to set this as the interpreter.
6. Open the PyCharm terminal window.
7. Ensure you are in the conda environment that you just created. If you are not, then type `conda activate environment_name`, where environtment_name is the name of the environment you created.
8. Enter the following command into the PyCharm terminal to install the dependencies: `pip install -r requirements.txt`
9. If any of the dependencies do not install correctly, then try to install them individually. For example, tensorflow can be installed by using `pip install tensorflow==2.7.0` or `pip install tensorflow --upgrade`
10. Wait for PyCharm to finish indexing.
11. Right click on the `src` folder, scroll down to ‘Mark Directory as’, and ensure it is marked as the sources root.
12. Open one of the files in the `mains` folder, such as `magicseaweed_binary.py`
13. Uncomment one of the functions inside of the `main()` function and run the file (in each of the files the `train_and_test_model()` function has already been uncommented)
