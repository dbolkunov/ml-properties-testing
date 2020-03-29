import operator

import pandas as pd
import pytest

from .rules import RuleValue

test_data = pd.read_csv("./data/test_data.csv", index_col=0)
RULES = [
    RuleValue(
        sample=test_data.loc["big_light_diamond"].to_dict(),
        comparator=operator.ge,
        value=0,
    ),
    RuleValue(
        sample=test_data.loc["small_light_diamond"].to_dict(),
        comparator=operator.ge,
        value=0,
    ),
]


@pytest.mark.parametrize("rule", RULES)
def test_rule_value(pipeline, rule: RuleValue):
    pred = pipeline.predict(pd.DataFrame([rule.sample]))[0]
    assert rule.comparator(pred, rule.value)
