"""Processors for the data cleaning step of the worklow.

The processors in this step, apply the various cleaning steps identified
during EDA to create the training datasets.
"""
import logging
import logging.config
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tarfile
from scripts import binned_median_income
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from housing.core.api import (
    custom_train_test_split,
    load_dataset,
    register_processor,
    save_dataset,
    string_cleaning,
)


@register_processor("data-cleaning", "housing")
def clean_housing_table(context, params,imputer=None):
    """Clean the ``HOUSING`` data table.

    The table containts the housing data and has information on the total_rooms,
    the median_income, the median_house_value etc.
    """


    print('''This job has 2 tasks:-
    1. Cleaning the raw data and store in processed folder
    2. Splitting the original data in train and test set ''')

    input_dataset = "raw/housing"
    output_dataset = "processed/housing"

    print('Task 1')

    # load dataset
    print('Loading original dataset')
    housing_df = load_dataset(context, input_dataset)

    # Imputing Missing Values
    print('Imputing Missing Values with help of Simple Imputer')
    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        housing_num = housing_df.drop("ocean_proximity", axis=1)
        imputer.fit(housing_num)

    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_df.index)

    print("Creation of new features from existing features")

    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    print("Treating Categorical Variable")

    housing_cat = housing_df[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    # save dataset
    print('Saving the processed dataset')
    save_dataset(context,housing_prepared,output_dataset)
    return housing_prepared


@register_processor("data-cleaning", "train-test")
def create_training_datasets(context, params):
    """Split the ``HOUSING`` table into ``train`` and ``test`` datasets."""

    print('Task 2')

    input_dataset = "raw/housing"
    output_train_features = "train/housing/features"
    output_train_target = "train/housing/target"
    output_test_features = "test/housing/features"
    output_test_target = "test/housing/target"

    # load dataset
    print("Loading the Original Dataset")
    housing_df_original= load_dataset(context, input_dataset)

    # split the data
    print("Creating the Test and Train Data from Original using StratifiedShuffling Method by median_income feature")
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=params["test_size"], random_state=context.random_seed
    )
    housing_df_train, housing_df_test = custom_train_test_split(
        housing_df_original, splitter, by=binned_median_income
    )

    # split train dataset into features and target
    print("Split train dataset into features and target")
    target_col = params["target"]
    train_X, train_y = (
        housing_df_train
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the train dataset
    print("Saving The train Dataset")
    save_dataset(context, train_X, output_train_features)
    save_dataset(context, train_y, output_train_target)

    # split test dataset into features and target
    print("Split test dataset into features and target")
    test_X, test_y = (
        housing_df_test
        # split the dataset to train and test
        .get_features_targets(target_column_names=target_col)
    )

    # save the datasets
    print('Saving The test Dataset')
    save_dataset(context, test_X, output_test_features)
    save_dataset(context, test_y, output_test_target)
