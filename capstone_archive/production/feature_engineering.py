"""Processors for the feature engineering step of the worklow.

The step loads original training data, processes the data for outliers,
missing values and any other cleaning steps based on business rules/intuition.

The trained pipeline and any artifacts are then saved to be used in
training/scoring pipelines.
"""
import logging
import numpy as np
import os.path as op
import pandas as pd
from category_encoders import TargetEncoder
from scripts import CombinedAttributesAdder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from housing.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    register_processor,
    save_dataset,
    save_pipeline,
)
from housing.data_processing.api import Outlier

logger = logging.getLogger(__name__)


@register_processor("feat-engg", "transform-features")
def transform_features(context, params):
    """Transform dataset to create training datasets."""
    print("This job contains 1 task of transforming train dataset to make it desirable for ML Model")

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load datasets
    print('Loading Train Datasets')
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    cat_columns = train_X.select_dtypes("object").columns
    num_columns = train_X.select_dtypes("number").columns

    # Treating Outliers
    print('Treating Outliers')
    outlier_transformer = Outlier(method=params["outliers"]["method"])
    train_X = outlier_transformer.fit_transform(
        train_X, drop=params["outliers"]["drop"]
    )




    # Craeting a numerical pipeline
    print('Creating Transformational Pipeline for Numerical Data')
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

    # Creating full pipeline
    print("Creating full pipeline")
    full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_columns),
            ("cat", OneHotEncoder(), cat_columns),
        ])


    # NOTE: You can use ``Pipeline`` to compose a collection of transformers
    # into a single transformer. In this case, we are composing a
    # ``OneHotEncoder`` and a ``numerical pipeline`` to first encode the
    # categorical variable into a numerical values and then process
    # numerical data


    # Train the feature engg. pipeline prepared earlier. Note that the pipeline is
    # fitted on only the **training data** and not the full dataset.
    # This avoids leaking information about the test dataset when training the model.
    # In the below code train_X, train_y in the fit_transform can be replaced with
    # sample_X and sample_y if required.

    print("Transforming Full features")

    train_X = get_dataframe(
        full_pipeline.fit_transform(train_X),
        get_feature_names_from_column_transformer(full_pipeline),
    )

    print('Transformation done')

    print("Saving modified train dataset")

    save_dataset(context, train_X, input_features_ds )

    print("Saving full pipeline for transformation")
    save_pipeline(
        full_pipeline, op.abspath(op.join(artifacts_folder, "features.joblib"))
    )
