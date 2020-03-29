import typing as T
from dataclasses import dataclass


# compare prediction for a sample with certain value
@dataclass
class RuleValue:
    # test sample
    sample: T.Dict
    # how we compare this (leq, geq, ...)
    comparator: T.Callable
    # value to compare
    value: float


# check that prediction value of one sample is greater than of another one
@dataclass
class RuleSamplePredGreater:
    sample: T.Dict
    sample_smaller: T.Dict
    # gap between prediction for smaller sample and bigger one
    gap: float
