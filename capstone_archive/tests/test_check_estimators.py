from sklearn.utils.estimator_checks import parametrize_with_checks

from production.scripts import CombinedAttributesAdder


@parametrize_with_checks([CombinedAttributesAdder()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
