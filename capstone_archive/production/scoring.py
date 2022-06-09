"""Processors for the model scoring/evaluation step of the worklow."""
import os.path as op

from housing.core.api import (
    DEFAULT_ARTIFACTS_PATH,
    get_dataframe,
    get_feature_names_from_column_transformer,
    get_package_path,
    hash_object,
    load_dataset,
    load_pipeline,
    register_processor,
    save_dataset,
)


@register_processor("model-eval", "score-model")
def score_model(context, params):
    """Score a pre-trained model."""

    print('This job contains 1 task of loading the training pipeline and using it to predict test targets')
    input_features_ds = "test/housing/features"
    input_target_ds = "test/housing/target"
    output_ds = "score/housing/output"

    artifacts_folder = DEFAULT_ARTIFACTS_PATH

    # load test datasets
    print('Loading test datasets')
    test_X = load_dataset(context, input_features_ds)
    test_y = load_dataset(context, input_target_ds)

    # load the feature pipeline and training pipelines
    print("Loading the feature pipeline and training pipelines")
    features_transformer = load_pipeline(op.join(artifacts_folder, "features.joblib"))
    model_pipeline = load_pipeline(op.join(artifacts_folder, "train_pipeline.joblib"))

    # transform the test dataset
    print("Transforming the test data using feature pipeline")
    test_X = get_dataframe(
        features_transformer.transform(test_X),
        get_feature_names_from_column_transformer(features_transformer),
    )


    # make a prediction
    print("Making prediction using training pipeline")
    test_X["yhat"] = model_pipeline.predict(test_X)
    print("Prediction are stored as 'yhat' column in score/scored_output.csv")

    # store the predictions for any further processing.
    test_X['y']=test_y
    save_dataset(context, test_X, output_ds)
    print('Saving prediction with original test_y and predicted test_yhat for comparison')
