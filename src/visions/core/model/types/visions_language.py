import pandas.api.types as pdt
import pandas as pd
import os

from visions.core.model.model_relation import relation_conf
from visions.core.model.models import VisionsBaseType


def get_language(series: pd.Series) -> str:
    import langdetect

    words = " ".join(series.values)
    return langdetect.detect(words)


def is_language(series: pd.Series):
    return all(s.isalpha() for s in series)


class visions_language(VisionsBaseType):
    """
    """

    @classmethod
    def get_relations(cls) -> dict:
        from visions.core.model.types import visions_string

        return {visions_string: relation_conf(inferential=False)}

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        from visions.core.model.types import visions_string

        if series not in visions_string:
            return False

        return is_language(series.dropna())


class visions_language_en(VisionsBaseType):
    @classmethod
    def get_relations(cls) -> dict:
        from visions.core.model.types import visions_generic

        return {visions_language: relation_conf(inferential=False)}

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        if series not in visions_language:
            return False

        return get_language(series.dropna()) == "en"


class visions_language_nl(VisionsBaseType):
    @classmethod
    def get_relations(cls) -> dict:
        from visions.core.model.types import visions_generic

        return {visions_language: relation_conf(inferential=False)}

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        if series not in visions_language:
            return False

        return get_language(series.dropna()) == "nl"


"""
supported_languages = []
for profile in os.listdir(langdetect.PROFILES_DIRECTORY):
    lang = profile.replace('-', '')
    language_name = ''.join(['visions_language_', lang])
    exec(f'{language_name} = language("{lang}")')
    supported_languages.append(language(lang))
"""
