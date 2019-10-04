from tenzing.core.model.types import *
from tenzing.core.model.model_relation import model_relation
from tenzing.utils import test_utils

from ipaddress import ip_address
from pathlib import Path, PureWindowsPath, PurePosixPath
from urllib.parse import urlparse
import numpy as np
import logging
import pandas as pd


def register_integer_relations():
    relations = [
        model_relation(tenzing_integer, tenzing_generic, inferential=False),
        model_relation(
            tenzing_integer,
            tenzing_float,
            inferential=True,
            relationship=test_utils.coercion_equality_test(lambda s: s.astype(int)),
        ),
        model_relation(
            tenzing_integer,
            tenzing_string,
            inferential=True,
            relationship=test_utils.coercion_test(lambda s: s.astype(float).astype("Int64")),
            transformer=lambda s: s.astype(float).astype("Int64"),
        ),
    ]

    return relations


def register_float_relations():
    def test_string_is_float(series):
        coerced_series = test_utils.option_coercion_evaluator(tenzing_float.cast)(
            series
        )
        return coerced_series is not None and coerced_series in tenzing_float

    relations = [
        model_relation(tenzing_float, tenzing_generic, inferential=False),
        model_relation(
            tenzing_float, tenzing_string, inferential=True, relationship=test_string_is_float
        ),
        model_relation(
            tenzing_float,
            tenzing_complex,
            inferential=True,
            relationship=lambda s: all(np.imag(s.values) == 0),
            transformer=lambda s: s.astype(float),
        )
    ]

    return relations


def register_string_relations():
    relations = [model_relation(tenzing_string, tenzing_object, inferential=False)]

    return relations


def register_url_relations():
    def test_url(series):
        try:
            return (
                series.apply(urlparse).apply(lambda x: all((x.netloc, x.scheme))).all()
            )
        except AttributeError:
            return False

    relations = [
        model_relation(tenzing_url, tenzing_string, relationship=test_url, inferential=True),
        model_relation(
            tenzing_url, tenzing_object, transformer=lambda s: s, inferential=False
        ),
    ]
    return relations


def register_path_relations():
    def string_is_path(series):
        try:
            s = series.copy().apply(PureWindowsPath)
            if not s.apply(lambda x: x.is_absolute()).all():
                return (
                    series.apply(PurePosixPath).apply(lambda x: x.is_absolute()).all()
                )
            else:
                return True
        # TODO: Too broad exception clause
        except Exception:
            return False

    relations = [
        model_relation(tenzing_path, tenzing_string, relationship=string_is_path, inferential=True),
        model_relation(tenzing_path, tenzing_object, inferential=False),
    ]
    return relations


def register_datetime_relations():
    relations = [
        model_relation(
            tenzing_datetime,
            tenzing_string,
            inferential=True,
            relationship=test_utils.coercion_test(lambda s: pd.to_datetime(s)),
        ),
        model_relation(
            tenzing_datetime,
            tenzing_integer,
            inferential=True,
            relationship=test_utils.coercion_test(lambda s: pd.to_datetime(s.astype(str))),
        ),
        model_relation(tenzing_datetime, tenzing_object, inferential=False),
        model_relation(tenzing_datetime, tenzing_generic, inferential=False),
    ]
    return relations


def register_timedelta_relations():
    relations = [
        model_relation(tenzing_timedelta, tenzing_object, inferential=False),
        model_relation(tenzing_timedelta, tenzing_generic, inferential=False),
    ]
    return relations


def register_geometry_relations():
    def string_is_geometry(series):
        """
            Shapely logs failures at a silly severity, just trying to suppress it's output on failures.
        """
        from shapely import wkt
        from shapely.errors import WKTReadingError

        logging.disable(logging.ERROR)
        try:
            result = all(wkt.loads(value) for value in series)
        except (WKTReadingError, AttributeError):
            result = False
        finally:
            logging.disable(logging.NOTSET)
        return result

    def string_to_geometry(series):
        from shapely import wkt
        return pd.Series([wkt.loads(value) for value in series])

    relations = [
        model_relation(
            tenzing_geometry, tenzing_string, inferential=True, relationship=string_is_geometry, transformer=string_to_geometry
        ),
        model_relation(
            tenzing_geometry,
            tenzing_object,
            inferential=False,
        ),
    ]
    return relations


def register_bool_relations():
    # Nullable bool: Object - > Bool
    class string_bool_relation:
        _boolean_maps = [
            {"true": True, "false": False},
            {"y": True, "n": False},
            {"yes": True, "no": False},
            {"j": True, "n": False},
            {"ja": True, "nee": False},
        ]

        def __init__(self):
            self._full_boolean_map = {
                k: v for d in self._boolean_maps for k, v in d.items()
            }

        # TODO: ensure that series.str.lower() has no side effects
        def string_is_bool(self, series):
            temp_series = series.str.lower()
            return any(
                temp_series.isin(boolean_map.keys()).all()
                for boolean_map in self._boolean_maps
            )

        def map_string_to_bool(self, series):
            return series.str.lower().map(self._full_boolean_map)

    sb_relation = string_bool_relation()
    relations = [
        model_relation(tenzing_bool, tenzing_generic, inferential=False),
        model_relation(
            tenzing_bool,
            tenzing_string,
            relationship=sb_relation.string_is_bool,
            transformer=sb_relation.map_string_to_bool,
            inferential=True,
        ),
        model_relation(
            tenzing_bool,
            tenzing_integer,
            relationship=lambda s: set(s.unique()) == {0, 1},
            inferential=True,
        )
    ]
    return relations


def register_categorical_relations():
    relations = [
        model_relation(tenzing_categorical, tenzing_generic, inferential=False),
        model_relation(tenzing_categorical, tenzing_string, inferential=True)
    ]
    return relations


def register_complex_relations():
    relations = [model_relation(tenzing_complex, tenzing_generic, inferential=False)]
    return relations


def register_object_relations():
    relations = [model_relation(tenzing_object, tenzing_generic, inferential=False)]
    return relations


def register_date_relations():
    relations = [model_relation(tenzing_date, tenzing_datetime, inferential=False)]
    return relations


def register_time_relations():
    relations = [model_relation(tenzing_time, tenzing_datetime, inferential=False)]
    return relations


def register_existing_path_relations():
    relations = [model_relation(tenzing_existing_path, tenzing_path, inferential=False)]
    return relations


def register_image_path_relations():
    relations = [
        model_relation(tenzing_image_path, tenzing_existing_path, inferential=False)
    ]
    return relations


def register_ip_relations():
    relations = [
        model_relation(tenzing_ip, tenzing_object, inferential=False),
        model_relation(
            tenzing_ip,
            tenzing_string,
            relationship=test_utils.coercion_test(lambda s: s.apply(ip_address)),
            inferential=True,
        ),
    ]
    return relations


def register_ordinal_relations():
    def check_consecutive(l):
        return sorted(l) == list(range(min(l), max(l) + 1))

    def is_ordinal_int(s):
        unique_values = s.unique()
        return check_consecutive(unique_values) and 2 < len(unique_values) < 10 and 1 in unique_values

    def is_ordinal_str(s):
        unique_values = s.str.lower().unique()
        return 'a' in unique_values and check_consecutive(list(map(ord, unique_values)))

    relations = [
        model_relation(tenzing_ordinal, tenzing_categorical, inferential=False),
        model_relation(tenzing_ordinal, tenzing_integer, relationship=is_ordinal_int, inferential=True),
        model_relation(tenzing_ordinal, tenzing_string, relationship=is_ordinal_str, inferential=True),
    ]
    return relations


# Register relations
relations = [
    register_integer_relations(),
    register_float_relations(),
    register_string_relations(),
    register_datetime_relations(),
    register_timedelta_relations(),
    register_bool_relations(),
    register_geometry_relations(),
    register_url_relations(),
    register_path_relations(),
    register_categorical_relations(),
    register_ordinal_relations(),
    register_complex_relations(),
    register_object_relations(),
    register_date_relations(),
    register_time_relations(),
    register_existing_path_relations(),
    register_ip_relations(),
    register_image_path_relations(),
]

for relation_list in relations:
    for relation in relation_list:
        relation.model.register_relation(relation)
