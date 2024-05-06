from raster_classes.S1Collection import S1Collection


class S1CompositeRaster(S1Collection):
    """ This class returns monthly sentinel-1 median composite for the given time period and AOI.
       Inputs:
       start: start date in string format
       end: end date in string format
       orbit pass: ASCENDING or DESCENDING
    """

    def __init__(self, start: str, end: str, orbit_pass: str = 'BOTH', kernel_size: int = 5,
                 no_of_images: int = 20, ratio: bool = True, preprocess=True):
        super().__init__(start, end, orbit_pass, kernel_size, no_of_images)
        self.instance_list.append(self)
        self.preprocess = preprocess
        # RasterClass identifier code
        self.code = 'S1CR'
        # satellite scale
        self.scale = 10
        self.orbit_pass = orbit_pass
        self.ratio = ratio
        self.band_ids = None
        self.processed = None
        self.raw = None

    def set_aoi(self, fc, province: str = None):
        """Sets the area of interest and triggers the actual computation of the raster class
        Inputs:
         fc: feature collection with one element, representing aoi
         province: name of province for storage purposes
        """
        super()._set_aoi(fc, province)
        if self.ratio:
            self.processed = self.slope_correction() \
                .map(self.add_ratio_lin) \
                .map(self.lin_to_db) \
                .select(['VV', 'VH', 'VHVV_ratio'])
            self.raw = self.get_s1_col() \
                .map(self.add_ratio_lin) \
                .map(self.lin_to_db) \
                .select(['VV', 'VH', 'VHVV_ratio'])
        else:
            self.processed = self.slope_correction().map(self.lin_to_db).select(['VV', 'VH'])
            self.raw = self.get_s1_col().map(self.lin_to_db).select(['VV', 'VH'])
        if self.preprocess:
            self.image = self.processed.median()
        else:
            self.image = self.raw.median()
        self.band_ids = self.get_feature_identifiers()
        return self
