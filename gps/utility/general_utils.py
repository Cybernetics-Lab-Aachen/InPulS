"""This file defines general utility functions and classes."""


class BundleType:
    """This class bundles many fields, similar to a record or a mutable namedtuple."""

    def __init__(self, variables):
        """Initializes the sample list.

        Args:
            variables: Dictionary of variable names and initial values.

        """
        for var, val in variables.items():
            object.__setattr__(self, var, val)

    def __setattr__(self, key, value):
        """Set an attribute if it's already present."""
        if not hasattr(self, key):  # Freeze fields so new ones cannot be set.
            raise AttributeError("%r has no attribute %s" % (self, key))
        object.__setattr__(self, key, value)


def check_shape(value, expected_shape, name=''):
    """Throws a ValueError if value.shape != expected_shape.

    Args:
        value: Matrix to shape check.
        expected_shape: A tuple or list of integers.
        name: An optional name to add to the exception message.

    """
    if value.shape != tuple(expected_shape):
        raise ValueError('Shape mismatch %s: Expected %s, got %s' % (name, str(expected_shape), str(value.shape)))


def extract_condition(hyperparams, m):
    """Pull the relevant hyperparameters corresponding to the specified condition.

    Args:
        hyperparams: Dictionary of hyperparameters.
        m: Condition

    Returns:
        new hyperparameter dictionary.

    """
    return {var: val[m] if isinstance(val, list) else val for var, val in hyperparams.items()}
