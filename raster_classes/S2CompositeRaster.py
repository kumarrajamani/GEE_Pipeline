from raster_classes.EERasterBuilder import EERasterBuilder
import ee
import math, datetime

class S2CompositeRaster(EERasterBuilder):
    def __init__(self,
                 start,
                 end,
                 bands=['B2', 'B3', 'B4', 'B8'],
                 cloud_filter=60,
                 cloudThresh=20,
                 irSumThresh=0.35,
                 dilatePixels=5,
                 contractPixels=1,
                 reducer='mosaic'):

        super().__init__(start, end)
        self.instance_list.append(self)
        # RasterClass identifier code
        self.code = 'S2CR'
        # satellite scale
        self.scale = 10
        # continue here with subclass specifics
        self.collection = 'COPERNICUS/S2_HARMONIZED'
        # included bands, defaults to rgb
        self.bands = bands
        # cloud filter parameters
        self.cloud_filter = cloud_filter
        # Cloud threshold
        # Ranges from 1-100.Lower value will mask more pixels out. Generally 10-30 works well with 20 being used most commonly
        self.cloudThresh = cloudThresh
        # IR sum threshold
        self.irSumThresh = irSumThresh
        # Pixels to dilate around clouds
        self.dilatePixels = dilatePixels
        # Pixels to reduce cloud mask and dark shadows by to reduce inclusion of single-pixel comission errors
        self.contractPixels = contractPixels
        # Height of clouds to use to project cloud shadows
        self.cloudHeights = ee.List.sequence(200,10000,500)
        # select reducer function
        self.reducer = reducer

    def get_s2_cld_col(self):
        """ Import and filter S2 SR. Import and filter s2cloudless.
        Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
        Import and filter S2 SR """

        s2_col = (ee.ImageCollection('COPERNICUS/S2')
                  .filterBounds(self.aoi)
                  .filterDate(self.start, self.end)
                  .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', self.cloud_filter)))

        s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
                            .filterBounds(self.aoi)
                            .filterDate(self.start, self.end))
        return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
            'primary': s2_col,
            'secondary': s2_cloudless_col,
            'condition': ee.Filter.equals(**{
                'leftField': 'system:index',
                'rightField': 'system:index'
            })
        }))

    def add_cloud_bands(self, img: ee.Image):
        """ Get s2cloudless image, subset the probability band.
        Condition s2cloudless by the probability threshold value.
        Add the cloud probability layer and cloud mask as image bands. """
        cld_prb = ee.Image(img.get('s2cloudless')).select('probability')
        is_cloud = cld_prb.gt(self.cloudThresh).rename('clouds')
        return img.addBands(ee.Image([cld_prb, is_cloud]))

    def apply_s2cloudless_mask(self, img):
        img = self.maskS2clouds(img)
        img = self.add_cloud_bands(img)
        not_cld = img.select('clouds').Not()

        # Subset reflectance bands and update their masks, return the result.
        return img.select('B.*').updateMask(not_cld)

    def set_aoi(self, fc, province=None):
        """Sets the area of interest and triggers the actual computation of the raster class
        Inputs:
         fc: feature collection with one element, representing aoi
         province: name of province for storage purposes
        """
        self.aoi = fc
        self.province = province
        # compute raster
        self.image = self.compute_raster()
        self.band_ids = self.get_feature_identifiers()
        return self



    def compute_raster(self):
        """Computes the cloud-free composite raster and returns an ee.Image"""

        s2_col = self.get_s2_cld_col()
        s2_col = s2_col.map(self.apply_s2cloudless_mask)
        s2_col = s2_col.map(self.rename_bands).map(self.wrapIt)

        if self.reducer == 'mean':
            s2_sr_image = s2_col.mean()
        elif self.reducer == 'max':
            s2_sr_image = s2_col.max()
        elif self.reducer == 'median':
            median = s2_col.median()
            def calculate_diff(img):
                dif = ee.Image(img).subtract(median).pow(ee.Image.constant(2));
                return dif.reduce(ee.Reducer.sum()).addBands(img).copyProperties(img, [
                    'system:time_start'
                ])
            difFromMedian = s2_col.map(calculate_diff)
            bandNames = difFromMedian.first().bandNames().getInfo()
            bandPositions = ee.List.sequence(1, len(bandNames)-1)
            s2_sr_image = difFromMedian.reduce(ee.Reducer.min(len(bandNames)))\
                .select(bandPositions, ee.List(bandNames[1:]))
        else:
            s2_sr_image = s2_col.mosaic()
        return s2_sr_image.select(self.bands).clip(self.aoi)

    def wrapIt(self, img):
        img = self.sentinelCloudScore(img)
        img = self.projectShadows(img)
        return img.select(['cb', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 'nir2', 'waterVapor', 'cirrus', 'swir1',
             'swir2'], ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'])

    def rescale(self, img, thresholds):
        """
        Linear stretch of image between two threshold values.
        """
        return img.subtract(thresholds[0]).divide(thresholds[1] - thresholds[0])

    def sentinelCloudScore(self, img):
        """
        Computes spectral indices of cloudyness and take the minimum of them.

        Each spectral index is fairly lenient because the group minimum
        is a somewhat stringent comparison policy. side note -> this seems like a job for machine learning :)

        originally written by Matt Hancher for Landsat imagery
        adapted to Sentinel by Chris Hewig and Ian Housman
        """

        # Compute several indicators of cloudyness and take the minimum of them.
        score = ee.Image(1)

        # Clouds are reasonably bright in the blue and cirrus bands.
        score = score.min(self.rescale(img.select(['blue']), [0.1, 0.5]))
        score = score.min(self.rescale(img.select(['cb']), [0.1, 0.3]))
        score = score.min(self.rescale(img.select(['cb']).add(img.select(['cirrus'])), [0.15, 0.2]))

        # Clouds are reasonably bright in all visible bands.
        score = score.min(self.rescale(img.select(['red']).add(img.select(['green'])).add(img.select('blue')), [0.2, 0.8]))

        # clouds are moist
        ndmi = img.normalizedDifference(['nir', 'swir1'])
        score = score.min(self.rescale(ndmi, [-0.1, 0.1]))

        # clouds are not snow.
        ndsi = img.normalizedDifference(['green', 'swir1'])
        score = score.min(self.rescale(ndsi, [0.8, 0.6]))

        score = score.multiply(100).byte()

        return img.addBands(score.rename(['cloudScore']))

    def maskS2clouds(self, img):
        qa = img.select('QA60')
        cloudBitMask = 1 << 10
        cirrusBitMask = 1 << 11
        mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        return img.updateMask(mask)

    def projectShadows(self, img):
        meanAzimuth = img.get('MEAN_SOLAR_AZIMUTH_ANGLE')
        meanZenith = img.get('MEAN_SOLAR_ZENITH_ANGLE')

        # Find dark pixels
        darkPixels = img.select(['nir', 'swir1', 'swir2']).reduce(ee.Reducer.sum()).lt(self.irSumThresh).focal_min(
            self.contractPixels).focal_max(self.dilatePixels)
        # Get scale of image
        cloudMask = img.select(['cloudScore']).gt(self.cloudThresh) \
            .focal_min(self.contractPixels).focal_max(self.dilatePixels)
        nominalScale = cloudMask.projection().nominalScale()

        # Find where cloud shadows should be based on solar geometry
        # Convert to radians

        azR = ee.Number(meanAzimuth).multiply(math.pi).divide(180.0).add(ee.Number(0.5).multiply(math.pi))
        zenR = ee.Number(0.5).multiply(math.pi).subtract(ee.Number(meanZenith).multiply(math.pi).divide(180.0))

        # Find the shadows

        def potentialShadow(cloudHeight):
            cloudHeight = ee.Number(cloudHeight)
            shadowCastedDistance = zenR.tan().multiply(cloudHeight);  # Distance shadow is cast
            x = azR.cos().multiply(shadowCastedDistance).divide(nominalScale).round();  # X distance of shadow
            y = azR.sin().multiply(shadowCastedDistance).divide(nominalScale).round();  # Y distance of shadow
            return cloudMask.changeProj(cloudMask.projection(), cloudMask.projection().translate(x, y))


        shadows = self.cloudHeights.map(potentialShadow)
        shadowMask = ee.ImageCollection.fromImages(shadows).max()

        # Create shadow mask
        shadowMask = shadowMask.And(cloudMask.Not())
        shadowMask = shadowMask.And(darkPixels)

        cloudShadowMask = shadowMask.Or(cloudMask)

        img = img.updateMask(cloudShadowMask.Not()).addBands(shadowMask.rename(['cloudShadowMask']))
        return img

    def rename_bands(self, img):
        t = img.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']).divide(10000)
        out = t.copyProperties(img).copyProperties(img, ['system:time_start'])
        return ee.Image(out).select(
            ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'],
            ['cb', 'blue', 'green', 'red', 're1', 're2', 're3', 'nir', 'nir2', 'waterVapor', 'cirrus', 'swir1',
             'swir2'])

