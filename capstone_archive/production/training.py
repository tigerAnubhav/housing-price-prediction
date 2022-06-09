"""Processors for the model training step of the worklow."""
import logging
import os.path as op
from sklearn.pipeline import Pipeline

from housing.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    load_dataset,
    load_pipeline,
    register_processor,
    save_pipeline,
)
from housing.regression.api import SKLStatsmodelOLS

logger = logging.getLogger(__name__)


@register_processor("model-gen", "train-model")
def train_model(context, params):
    """Train a regression model."""
    print("This job contains 1 task of creating a regression model")
    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    input_features_ds = "train/housing/features"
    input_target_ds = "train/housing/target"

    # load training datasets which are already passed through piplelines for feature engineering
    print("Load training datasets which are already passed through piplelines for feature engineering")
    train_X = load_dataset(context, input_features_ds)
    train_y = load_dataset(context, input_target_ds)

    # create training pipeline
    print("Create training pipeline")
    reg_ppln_ols = Pipeline([("estimator", SKLStatsmodelOLS())])

    # fit the training pipeline
    print("Fitting the training pipeline")
    reg_ppln_ols.fit(train_X, train_y.values.ravel())

    # save fitted training pipeline
    print("Saving fitted training pipeline")
    save_pipeline(
        reg_ppln_ols, op.abspath(op.join(artifacts_folder, "train_pipeline.joblib"))
    )
