import pandas as pd
import pytest

from src.train_model import train_model, COLUMN_TARGET, ML_MODELS_AVAILABLE


@pytest.fixture(
    # from python 3.7 dicts are ordered
    params=list(ML_MODELS_AVAILABLE.values()),
    ids=list(ML_MODELS_AVAILABLE.keys()),
)
def pipeline(request):
    df = pd.read_csv("./data/diamonds.csv", index_col=0)
    model = train_model(
        df.drop(COLUMN_TARGET, axis=1), df[COLUMN_TARGET], model=request.param
    )
    return model
