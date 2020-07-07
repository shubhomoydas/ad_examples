import numpy as np
import pandas as pd
from .expressions import (
    NumericContinuous, Factor, FeatureMetadata, string_to_predicate, evaluate_instances_for_predicate
)

"""
Some very simple rule/predicate expression examples have been presented 
so that the interested reader can start using the light-weight rule-processing
framework.

Pandas is extremely feature-rich, but has performance issues. The intent
is to use the numpy arrays/matrices for most of our processing, and TO
employ our rule-processor when needed for rule-based abstractions.

The most crucial data structure is the FeatureMetadata.

To run:
python -m ad_examples.common.expressions_tutorial
"""


csv_contents = (
    "104,F,30,150,-1\n"
    "509,M,45,169,-3\n"
    "423,F,28,156,2"
    )


def load_data():
    # Handle both python 2.7 and 3.6
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO
    # print(csv_contents)
    stream = StringIO(csv_contents)
    df = pd.read_csv(stream, sep=',', delimiter=None, header=None)
    return df


def dataframe_to_numpy(df, meta):
    """ Converts Pandas dataframe to numpy array

    This handles both numerical and categorical data while loading
    into numpy array. Processing with numpy arrays is faster than
    processing with Pandas. Hence we apply this conversion and use
    our light-weight rule processor on top of the numpy array.
    """
    mat = np.zeros(shape=df.shape, dtype=np.float32)
    for i in range(df.shape[1]):
        fdef = meta.featuredefs[i]
        if isinstance(fdef, NumericContinuous):
            mat[:, i] = df.iloc[:, i]
        elif isinstance(fdef, Factor):
            ivals = [fdef.index_of(v) for v in df.iloc[:, i]]
            mat[:, i] = ivals
    return mat


if __name__ == "__main__":

    # Define feature metadata that can handle both categorical
    # and numeric features.
    #
    # NOTE: For featurenames, Factor *MUST* have sort=False
    # so that feature order is maintained.
    # For numerical features, it does not matter which values we
    # pass to constructor of NumericContinuous. We just pass [0, 1]
    meta = FeatureMetadata(featurenames=Factor(["id", "sex", "age", "weight", "score"], sort=False),
                           featuredefs=[NumericContinuous(),  # id
                                        Factor(["M", "F"]),  # sex
                                        NumericContinuous(),  # age
                                        NumericContinuous(),  # weight
                                        NumericContinuous()  # score
                                        ]
                           )
    print(str(meta))

    df = load_data()
    print("\nPandas dataframe:")
    print(df)

    insts = dataframe_to_numpy(df, meta=meta)
    # Values for the categorical feature 'sex' should have
    # got converted into numeric values.
    print("\nData loaded into numpy matrix:")
    print(insts)

    # a few sample rules to demonstrate rule evaluation
    str_rules = [
        "sex = 'F'",
        "weight > 150",
        "age < 30 & sex = 'F'",
        "score > -1",
        "~(weight > 160) | (sex = 'M')"  # Implication: (weight > 160) => (sex = 'M')
    ]

    for str_rule in str_rules:
        rule = string_to_predicate(str_rule, meta=meta)
        # print("\nRule: %s" % str(rule))
        print("\nRule: %s" % rule.expr(meta))
        sat = evaluate_instances_for_predicate(predicate=rule, insts=insts, labels=None, meta=meta)
        print("Instances that satisfy: %s" % str(sat))
