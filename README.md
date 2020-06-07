## Testing ML Model Properties

This repo contains an example of testing properties of ML model with the help of pytest.

In essence it's testing ML predictions on corner cases to make sure that results of the model are 
reasonable. For example, one of the tests here is that predicted diamond price should be greater than
zero. And you can see that with even very simple model it can be violated.

This approach can help to find
1. bugs in the model (small failed test can give you intuition what went wrong)
2. biasses in the train dataset
3. flaws of used metric

More details can be found in this [article](https://medium.com/@d.bolkunov/testing-properties-of-machine-learning-models-1b9745bca48e).

## Installation
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2. Install conda environment
    ```bash
    conda env update -f environment.yml
    ```
3. Activate conda environment
    ```bash
    conda activate ml-properties-testing
    ```

## How To

#### Test properties of ML Model
```bash
pytest tests
```

If you run these, you'll see that `ridge` model has 2 failed tests, showing that this model predicts
negative prices for light diamonds. However, `ridge_target_log_transformed` passes all tests.

#### Train and Evaluate Given Model
```bash
python -m src.train_model --path data/diamonds.csv --model-name ridge
```
Instead of `ridge` model you can also use `ridge_target_log_transformed`.
