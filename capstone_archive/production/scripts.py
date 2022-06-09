"""Module for listing down additional custom functions required for production."""

import numpy as np
import pandas as pd


def binned_median_income(housing_df):
    """Bin the selling price column using quantiles."""
    return pd.cut(housing_df["median_income"], bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1, 2, 3, 4, 5])

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

    #Creating a Custom Transform for addition of additional attributes like
    # rooms_per_household, population_per_household,bedrooms_per_room

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    """ Creating my own estimator for addtion of additional features from existing features.
         Additional features created inclue-
         1. rooms_per_household
         2. population_per_household
         3. bedrooms_per_room"""

    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                            bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
