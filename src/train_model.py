import json
import logging

import click
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .utils import (
    create_feature_transformers,
    mean_absolute_percentage_error,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger()

COLUMN_TARGET = "price"
COLUMNS_CATEGORICAL = ["cut", "color", "clarity"]
ML_MODELS_AVAILABLE = {
    "ridge": Ridge(alpha=1),
    "ridge_target_log_transformed": TransformedTargetRegressor(
        regressor=Ridge(), func=np.log, inverse_func=np.exp
    ),
}


def train_model(xtrain, ytrain, model):
    transformers = create_feature_transformers(
        categorical_columns=COLUMNS_CATEGORICAL,
        other_columns=xtrain.columns.drop(COLUMNS_CATEGORICAL),
    )

    pipeline = Pipeline([("transform", transformers), ("model", model)])

    pipeline = pipeline.fit(xtrain, ytrain)
    return pipeline


def estimate_model(ypred, ytest):
    return {
        "mae": mean_absolute_error(ypred, ytest),
        "rmse": np.sqrt(mean_squared_error(ypred, ytest)),
        "mape": mean_absolute_percentage_error(ypred, ytest),
    }


@click.command()
@click.option("--path", type=click.STRING, help="path to csv file with data")
@click.option(
    "--model-name",
    type=click.Choice(list(ML_MODELS_AVAILABLE.keys())),
    help="choose ml model",
)
def main(path: str, model_name: str):
    df = pd.read_csv(path, index_col=0)
    xtrain, xtest, ytrain, ytest = train_test_split(
        df.drop(COLUMN_TARGET, axis=1), df[COLUMN_TARGET], test_size=0.1, shuffle=True
    )
    model = ML_MODELS_AVAILABLE[model_name]

    pipeline = train_model(xtrain, ytrain, model)
    ypred = pipeline.predict(xtest)
    metrics = estimate_model(ypred, ytest,)
    logger.info(f"Metrics for the model {model_name}: {json.dumps(metrics, indent=2)}")

    ypred_mean = np.array([ytrain.mean()] * ytest.shape[0])
    metrics_mean = estimate_model(ypred_mean, ytest)
    logger.info(f"Metrics mean prediction: {json.dumps(metrics_mean, indent=2)}")


if __name__ == "__main__":
    main()
