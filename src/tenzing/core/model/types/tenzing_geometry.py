import pandas as pd

from tenzing.core.model.models import tenzing_model


# https://jorisvandenbossche.github.io/blog/2019/08/13/geopandas-extension-array-refactor/
class tenzing_geometry(tenzing_model):
    """**Geometry** implementation of :class:`tenzing.core.models.tenzing_model`.
    >>> from shapely import wkt
    >>> x = pd.Series([wkt.loads('POINT (-92 42)'), wkt.loads('POINT (-92 42.1)'), wkt.loads('POINT (-92 42.2)')]
    >>> x in tenzing_geometry
    True
    """

    @classmethod
    def contains_op(cls, series: pd.Series) -> bool:
        from shapely.geometry.base import BaseGeometry
        return series.apply(lambda x: issubclass(type(x), BaseGeometry)).all()
        # The below raises `TypeError: data type "geometry" not understood`
        # import geopandas
        # from geopandas import array
        # from geopandas.array import GeometryDtype
        # return series.dtype == geometry

    @classmethod
    def cast_op(cls, series: pd.Series, operation=None) -> pd.Series:
        import geopandas
        return geopandas.GeoSeries(series.values)
        # from shapely import wkt
        # return pd.Series([wkt.loads(value) for value in series])
