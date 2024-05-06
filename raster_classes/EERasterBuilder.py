import ee
from datetime import datetime
from itertools import groupby


class EERasterBuilder:
    instance_list = []

    def __init__(self, start: str = '2020-01-01', end: str = '2021-01-31', reducer_fc: str = 'mean'):
        self.start = start
        self.end = end
        self.start_date = datetime.fromisoformat(self.start)
        self.end_date = datetime.fromisoformat(self.end)
        self.date_code = self.start.replace('-', '') + self.end.replace('-', '')
        self._tf = '%Y-%m-%d'
        self.image = None
        # reducer string that will be used in Feature Extraction
        self.reducer_fc = reducer_fc

    def get_feature_identifiers(self):
        """ Returns a list of feature identifiers and renames image bands using this list """
        # load the original image band names
        bands = self.image.bandNames().getInfo()
        # build new band names that identify the raster class, time, and band
        band_ids = [self.code + self.date_code + name for name in bands]
        # rename the image bands
        self.image = self.image.rename(band_ids)
        return band_ids

    def _nearest(self, items: list, pivot: str):
        """ Returns the closest date from a list to a given date
        Inputs:
            items: list of dates
            pivot: anchor date"""
        return min(items, key=lambda x: abs(x - pivot))

    def _all_equal(self, iterable):
        """ Evaluates if all elements of iterable are identical"""
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    def _set_aoi(self, fc: ee.FeatureCollection, province: str = None):
        """ Assign fc anf province values, this function is called in all inherited classes."""
        self.aoi = fc
        self.province = province
