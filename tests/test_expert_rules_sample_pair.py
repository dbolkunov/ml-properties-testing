import pandas as pd
import pytest

from .rules import RuleSamplePredGreater

test_data = pd.read_csv("./data/test_data.csv", index_col=0)

RULES = [
    RuleSamplePredGreater(
        sample=test_data.loc["big_diamond_ideal_cut"].to_dict(),
        sample_smaller=test_data.loc["big_diamond_good_cut"].to_dict(),
        gap=100,  # at least 100 dollars for better cut
    ),
    RuleSamplePredGreater(
        sample=test_data.loc["big_diamond_ideal_cut"].to_dict(),
        sample_smaller=test_data.loc["big_diamond_fair_cut"].to_dict(),
        gap=300,  # at least 300 dollars for much better cut
    ),
    RuleSamplePredGreater(
        sample=test_data.loc["big_diamond_ideal_cut"].to_dict(),
        sample_smaller=test_data.loc["big_diamond_ideal_cut_smaller"].to_dict(),
        gap=10,  # at least 10 dollars for smaller diamond
    ),
]


@pytest.mark.parametrize("rule", RULES)
def test_rule_value(pipeline, rule: RuleSamplePredGreater):
    pred = pipeline.predict(pd.DataFrame([rule.sample]))[0]
    pred_smaller = pipeline.predict(pd.DataFrame([rule.sample_smaller]))[0]
    assert (pred - pred_smaller) > rule.gap
