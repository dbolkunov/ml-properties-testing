import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def create_feature_transformers(categorical_columns, other_columns):
    onehot_tranformed_cols = make_column_transformer(
        *[(OneHotEncoder(sparse=False), [col]) for col in categorical_columns]
    )

    IdentetyTransformer = FunctionTransformer(lambda x: np.log1p(x), validate=True)

    identity_transformed_cols = make_column_transformer(
        *[(IdentetyTransformer, [col]) for col in other_columns]
    )

    transformers = FeatureUnion(
        [("onehot", onehot_tranformed_cols), ("identity", identity_transformed_cols)]
    )
    return transformers


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
