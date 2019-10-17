import pandas.api.types as pdt
import pandas as pd
import numpy as np

from visions.core.model.model_relation import relation_conf
from visions.core.model.models import VisionsBaseType
from visions.utils.coercion.test_utils import coercion_map_test, coercion_map


def to_bool(series: pd.Series) -> pd.Series:
    if series.isin({True, False}).all():
        return series.astype(bool)
    elif series.isin({True, False, None, np.nan}).all():
        return series.astype("Bool")
    else:
        unsupported_values = series[series.isin({True, False, None, np.nan})].unique()
        raise ValueError(f"Values not supported {unsupported_values}")


def get_language_coercions(language_code):
    lang_map = {
        "en": [
            {"true": True, "false": False},
            {"y": True, "n": False},
            {"yes": True, "no": False},
        ],
        "nl": [{"ja": True, "nee": False}],
    }
    return lang_map[language_code]


class visions_bool(VisionsBaseType):
    """**Boolean** implementation of :class:`visions.core.models.VisionsBaseType`.
    >>> x = pd.Series([True, False, False, True])
    >>> x in visions_bool
    True

    >>> x = pd.Series([True, False, None])
    >>> x in visions_bool
    True
    """

    string_coercions = get_language_coercions("en")

    @classmethod
    def get_relations(cls) -> dict:
        from visions.core.model.types import (
            visions_generic,
            visions_string,
            visions_integer,
            visions_object,
        )

        relations = {
            visions_generic: relation_conf(inferential=False),
            # TODO: ensure that series.str.lower() has no side effects
            visions_string: relation_conf(
                inferential=True,
                relationship=lambda s: coercion_map_test(cls.string_coercions)(
                    s.str.lower()
                ),
                transformer=lambda s: to_bool(
                    coercion_map(cls.string_coercions)(s.str.lower())
                ),
            ),
            visions_integer: relation_conf(
                inferential=True,
                relationship=lambda s: s.isin({0, 1, np.nan}).all(),
                transformer=to_bool,
            ),
            visions_object: relation_conf(
                inferential=True,
                relationship=lambda s: s.apply(type).isin([type(None), bool]).all(),
                transformer=to_bool,
            ),
        }
        return relations

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        return not pdt.is_categorical_dtype(series) and pdt.is_bool_dtype(series)

    @classmethod
    def set_string_coercion_by_language(cls, language_code="en"):
        cls.set_string_coercion(get_language_coercions(language_code))

    @classmethod
    def set_string_coercion(cls, string_coercions):
        cls.string_coercions = string_coercions
