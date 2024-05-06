from raster_classes.EERasterBuilder import EERasterBuilder
from raster_classes.S2CompositeRaster import S2CompositeRaster
import ee



class S2index(EERasterBuilder):
    """
    Compute S2 indices for the given S2-image
    Inputs:
        S2 Image raster class object with 10 bands ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        choose one of the indices from the list  ['NDVI', 'NDWI', 'GCVI', 'MNDWI', 'NDBI', 'BSI',
                                                  'EVI', 'NDVI_GRAD','REP', 'CI_re']
    Usage:
        ndvi = S2index(S2_img, ['NDVI']).image
        evi = S2index(S2_img, ['EVI']).image
        gcvi = S2index(S2_img, ['GCVI']).image
        ndwi = S2index(S2_img, ['NDWI']).image
        mndwi = S2index(S2_img, ['MNDWI']).image
        bsi = S2index(S2_img, ['BSI']).image
        ndbi = S2index(S2_img, ['NDBI']).image
        ndvi_grad = S2index(S2_img, ['NDVI_GRAD']).image
        red_edge = S2index(S2_img, ['REP']).image
    """

    def __init__(self, s2rc: S2CompositeRaster, index: list):
        super().__init__()
        self.instance_list.append(self)
        self.parent = s2rc
        self.index = index
        self.code = 'S2IN'
        self.date_code = self.parent.date_code
        self.start = self.parent.start
        self.end = self.parent.end
        # satellite scale
        self.scale = 10
        self.band_mapping = None
        self.band_ids = None

    def set_aoi(self, fc: ee.FeatureCollection, province: str = None):
        """Sets the area of interest and triggers the actual computation of the raster class
        Inputs:
         fc: feature collection with one element, representing aoi
         province: name of province for storage purposes
        """
        super()._set_aoi(fc, province)
        self.parent.set_aoi(self.aoi, self.province)
        self.image = self.parent.image
        self.band_mapping = dict(zip(self.parent.bands, self.parent.band_ids))
        if self.index == ['ALL']:
            self.image = self.add_all()
        elif all(item in ['NDVI', 'NDRE', 'NDWI', 'MNDWI', 'NDBI', 'BSI', 'EVI', 'SAVI',
                          'NDVI_GRAD', 'REP', 'GCVI', 'LSWI', 'NDMI', 'CI_re']
                 for item in self.index):
            image_list = []
            for idx in self.index:
                if idx == 'NDVI':
                    image_list.append(self.add_ndvi())
                elif idx == 'NDWI':
                    image_list.append(self.add_ndwi())
                elif idx == 'MNDWI':
                    image_list.append(self.add_mndwi())
                elif idx == 'NDBI':
                    image_list.append(self.add_ndbi())
                elif idx == 'BSI':
                    image_list.append(self.add_bsi())
                elif idx == 'EVI':
                    image_list.append(self.add_evi())
                elif idx == 'SAVI':
                    image_list.append(self.add_savi())
                elif idx == 'NDVI_GRAD':
                    image_list.append(self.add_ndvi_gradient())
                elif idx == 'REP':
                    image_list.append(self.add_red_edge())
                elif idx == 'GCVI':
                    image_list.append(self.add_gcvi())
                elif idx == 'LSWI':
                    image_list.append(self.add_lswi())
                elif idx == 'NDMI':
                    image_list.append(self.add_ndmi())
                elif idx == 'CI_re':
                    image_list.append(self.add_ci_re())
                elif idx == 'NDRE':
                    image_list.append(self.add_ndre())
            self.image = ee.Image(image_list)

        else:
            raise ValueError(
                "Pass any of 'NDVI','NDWI', 'MNDWI', 'NDBI', 'BSI', 'EVI', 'SAVI',"
                " 'NDVI_GRAD', 'REP', 'GCVI', 'NBR', 'NDMI','CI_re'")
        self.band_ids = self.get_feature_identifiers()
        return self

    def get_feature_identifiers(self):
        """ Returns a list of feature identifiers and renames image bands using this list """
        # load the original image band names
        bands = self.image.bandNames().getInfo()
        # build new band names that identify the raster class, time, and band
        band_ids = [self.code + self.date_code + name for name in bands]
        # rename the image bands
        self.image = self.image.rename(band_ids)
        return band_ids

    def add_ndvi(self):
        """
        Returns Normalized Difference Vegetation Index (NDVI)
        General formula: (NIR - RED) / (NIR + RED)
        """
        return self.image.normalizedDifference([self.band_mapping['B8'], self.band_mapping['B4']]).rename('NDVI')

    def add_ndre(self):
        """
        Returns Normalized Difference Red Edge (NDRE)
        General formula: (NIR - RED EDGE) / (NIR + RED EDGE)
        """
        return self.image.normalizedDifference([self.band_mapping['B8'], self.band_mapping['B8A']]).rename('NDRE')

    def add_ndvi(self):
        """
        Returns Normalized Difference Vegetation Index (NDVI)
        General formula: (NIR - RED) / (NIR + RED)
        """
        return self.image.normalizedDifference([self.band_mapping['B8'], self.band_mapping['B4']]).rename('NDVI')

    def add_gcvi(self):
        """
        Returns  Green Chlorophyll Vegetation Index (GCVI)
        General formula: General formula: (NIR) / (GREEN - 1)
        """
        gcvi_img = self.image.expression('float(NIR / GREEN - 1)',
                                         {
                                             'NIR': self.image.select(self.band_mapping['B8']),
                                             'GREEN': self.image.select(self.band_mapping['B3'])
                                         }).rename('GCVI')
        return gcvi_img

    def add_ci_re(self):
        """
        Returns  red-edge chlorophyll index (CI_red_edge)
        General formula: General formula: (B7) / (B5) - 1
        """
        ci_img = self.image.expression('float(Red_Edge3 / Red_Edge1 - 1)',
                                       {
                                           'Red_Edge3': self.image.select(self.band_mapping['B7']),
                                           'Red_Edge1': self.image.select(self.band_mapping['B5'])
                                       }).rename('CI_re')
        return ci_img

    def add_ndwi(self):
        """
        Returns Normalized Difference Water Index (NDWI)
        General formula: General formula: (GREEN - NIR) / (GREEN + NIR)
        """
        return self.image.normalizedDifference([self.band_mapping['B3'], self.band_mapping['B8']]).rename(['NDWI'])

    def add_mndwi(self):
        """
        Returns Modified Normalized Difference Water Index (MNDWI)
        General formula: General formula: (GREEN - SWIR1) / (GREEN + SWR1)
        """
        return self.image.normalizedDifference([self.band_mapping['B3'], self.band_mapping['B11']]).rename(['MNDWI'])

    def add_ndbi(self):
        """
        Returns Normalized Difference Built-up Index (NDBI)
        General formula: General formula: (SWIR1 - NIR) / (SWIR1 + NIR)
        """
        return self.image.normalizedDifference([self.band_mapping['B11'], self.band_mapping['B8']]).rename(['NDBI'])

    def add_ndmi(self):
        """
        Returns Normalized Difference Moisture Index (NDMI)
        General formula: General formula: (Red Edge 4 - SWIR1) / (Red Edge 4 + SWIR1)
        """
        return self.image.normalizedDifference([self.band_mapping['B8A'], self.band_mapping['B11']]).rename(['NDMI'])

    def add_lswi(self):
        """
        Returns LSWI
        General formula: (NIR - SWIR) / (NIR + SWIR)
        """
        return self.image.normalizedDifference([self.band_mapping['B8'], self.band_mapping['B12']]).rename('LSWI')

    def add_bsi(self):
        """
        Returns Bare Soil Index (BSI)
        General formula: BSI = ((SWIR + RED)-(NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))
        """
        bsi_img = self.image.expression('float((SWIR + RED)-(NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
                                        {
                                            'SWIR': self.image.select(self.band_mapping['B11']),
                                            'RED': self.image.select(self.band_mapping['B4']),
                                            'NIR': self.image.select(self.band_mapping['B8']),
                                            'BLUE': self.image.select(self.band_mapping['B2'])
                                        }).rename('BSI')
        return bsi_img

    def add_evi(self):
        """
        Returns Enhanced Vegetation Index (EVI)
        General formula: 2.5 * (NIR - RED) / ((NIR + 6*RED - 7.5*BLUE) + 1)

        """
        evi_img = self.image.expression('float(2.5 * (NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
                                        {
                                            'RED': self.image.select(self.band_mapping['B4']),
                                            'NIR': self.image.select(self.band_mapping['B8']),
                                            'BLUE': self.image.select(self.band_mapping['B2'])
                                        }).rename('EVI')
        return evi_img

    def add_savi(self):
        """
        Returns Soil-Adjusted Vegetation Index (SAVI)
        General formula: (1 + L) * (NIR - RED) / (NIR + RED + L)
        where L = 0.48
        """
        savi_img = self.image.expression('float((1 + 0.48) * (NIR - RED)/(NIR + RED + 0.48))',
                                         {
                                             'RED': self.image.select(self.band_mapping['B4']),
                                             'NIR': self.image.select(self.band_mapping['B8'])
                                         }).rename('SAVI')
        return savi_img

    def add_ndvi_gradient(self):
        """
        Compute the magnitude of the NDVI gradient image
        """
        return self.add_ndvi().gradient().pow(2).reduce('sum').sqrt().rename('NDVI_GRAD')

    def add_red_edge(self):
        """
        Returns Red-Edge Position Linear Interpolation index (REP)
        General formula: 700+40*(((670nm+780nm)/2)-700nm)/(740nm-700nm))
        """

        rep_img = self.image.expression(
            'float(700 + 40 * ((((RED + RED_EDGE3) / 2) - RED_EDGE1)/(RED_EDGE2-RED_EDGE1)))',
            {
                'RED': self.image.select(self.band_mapping['B4']),
                'NIR': self.image.select(self.band_mapping['B8']),
                'BLUE': self.image.select(self.band_mapping['B2']),
                'RED_EDGE3': self.image.select(self.band_mapping['B7']),
                'RED_EDGE2': self.image.select(self.band_mapping['B6']),
                'RED_EDGE1': self.image.select(self.band_mapping['B5'])
            }).rename('REP')
        return rep_img

    def add_all(self):
        """
        Return all the indicators
        """
        return self.add_ndvi() \
            .addBands(self.add_ndre()) \
            .addBands(self.add_evi()) \
            .addBands(self.add_ndbi()) \
            .addBands(self.add_ndwi()) \
            .addBands(self.add_mndwi()) \
            .addBands(self.add_red_edge()) \
            .addBands(self.add_gcvi()) \
            .addBands(self.add_bsi()) \
            .addBands(self.add_lswi()) \
            .addBands(self.add_ndvi_gradient()) \
            .addBands(self.add_ndmi()) \
            .addBands(self.add_ci_re())
