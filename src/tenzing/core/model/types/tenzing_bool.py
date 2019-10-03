import pandas.api.types as pdt
import pandas as pd

from tenzing.core.model.models import tenzing_model


class tenzing_bool(tenzing_model):
    """**Boolean** implementation of :class:`tenzing.core.models.tenzing_model`.
    >>> x = pd.Series([True, False, None])
    >>> x in tenzing_bool
    True
    """

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        if pdt.is_categorical_dtype(series):
            return False

        return pdt.is_bool_dtype(series) or series.dtype == "Bool"

    @classmethod
    def cast_op(cls, series: pd.Series, operation=None) -> pd.Series:
        try:
            return series.astype(bool)
        except ValueError:
            return pd.to_numeric(series).astype("Bool")
