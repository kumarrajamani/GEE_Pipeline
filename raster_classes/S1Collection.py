from raster_classes.EERasterBuilder import EERasterBuilder
import ee
import math


class S1Collection(EERasterBuilder):
    def __init__(self, start: str, end: str, orbit_pass: str = 'BOTH', kernel_size: int = 7,
                 no_of_images: int = 20, model='VOLUME'):
        """ This class returns preprocessed S1-collection. Not for Feature Extraction.
            Inputs:
                orbit_pass: string ASCENDING or DESCENDING or BOTH
                kernel size: positive odd integer Neighbourhood window size
                no_of_images: positive integer  Number of images to use in multi-temporal filtering
                model: string The radiometric terrain normalization model, either VOLUME or DIRECT
        """
        super().__init__(start, end)
        self.orbit_pass = orbit_pass
        self.kernel_size = kernel_size
        self.no_of_images = no_of_images
        self.dem = ee.Image('USGS/SRTMGL1_003')
        self.terrain_flattening_model = model
        # additional buffer parameter for passive layover/shadow mask in meters
        self.terrain_flattening_additional_layover_shadow_buffer = 0

    def lin_to_db(self, image: ee.Image):
        """ Convert backscatter from linear to dB. """
        bandNames = image.bandNames().remove('angle')
        db = ee.Image.constant(10).multiply(image.select(bandNames).log10()).rename(bandNames)
        return image.addBands(db, None, True)

    def db_to_lin(self, image: ee.Image):
        """ Convert backscatter from dB to linear."""
        bandNames = image.bandNames().remove('angle')
        lin = ee.Image.constant(10).pow(image.select(bandNames).divide(10)).rename(bandNames)
        return image.addBands(lin, None, True)

    def lin_to_db2(self, image: ee.Image):
        """ Convert backscatter from linear to dB by removing the ratio band. """
        db = ee.Image.constant(10).multiply(image.select(['VV', 'VH']).log10()).rename(['VV', 'VH'])
        return image.addBands(db, None, True)

    def add_ratio_lin(self, image: ee.Image):
        """ Adding ratio band to the collection """
        ratio = image.addBands(image.select('VH').divide(image.select('VV')).rename('VHVV_ratio'))
        return ratio.set('system:time_start', image.get('system:time_start'))

    def get_s1_col(self):
        """ DATA SELECTION """
        if self.orbit_pass == 'BOTH':
            s1_col = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filterBounds(self.aoi).filterDate(self.start, self.end) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.eq('resolution_meters', 10)) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        else:
            s1_col = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                .filterBounds(self.aoi).filterDate(self.start, self.end) \
                .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                .filter(ee.Filter.eq('orbitProperties_pass', self.orbit_pass)) \
                .filter(ee.Filter.eq('resolution_meters', 10)) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        return s1_col

    def border_noise_correction(self):
        """ ADDITIONAL BORDER NOISE CORRECTION
            Function to mask out border noise artefacts
        """
        def maskAngLT452(image: ee.Image):
            """ Mask out angles >= 45.23993, returns masked image """
            ang = image.select(['angle'])
            return image.updateMask(ang.lt(45.23993)).set('system:time_start', image.get('system:time_start'))

        def maskAngGT30(image: ee.Image):
            """ Mask out angles <= 30.63993, returns masked image """
            ang = image.select(['angle'])
            return image.updateMask(ang.gt(30.63993)).set('system:time_start', image.get('system:time_start'))

        def f_mask_edges(image: ee.Image):
            """ Function to mask out border noise artefacts. Returns corrected image"""
            db_img = self.lin_to_db(image)
            output = maskAngGT30(db_img)
            output = maskAngLT452(output)
            output = self.db_to_lin(output)
            return output.set('system:time_start', image.get('system:time_start'))
        s1_col = self.get_s1_col().map(f_mask_edges)
        return s1_col

    def multi_temporal_filter(self):
        """
        A wrapper function for multi-temporal filter
        Parameters
            coll : ee Image collection the image collection to be filtered
            KERNEL_SIZE : odd integer, spatial neighbourhood window
            NR_OF_IMAGES : positive integer, number of images to use in multi-temporal filtering
        Returns
        ee.ImageCollection: An image collection where a multi-temporal filter is applied to each image individually
        """
        def boxcar(image: ee.Image, kernel_size):
            """
            Apply boxcar filter on every image in the collection.
            Parameters:
            image : ee.Image, the Image to be filtered
            kernel_size : positive odd integer, neighbourhood window size
            Returns:
            ee.Image: filtered Image
            """
            bandNames = image.bandNames().remove('angle')
            # Define a boxcar kernel
            kernel = ee.Kernel.square((kernel_size / 2), units='pixels', normalize=True)
            # Apply boxcar
            output = image.select(bandNames).convolve(kernel).rename(bandNames)
            return image.addBands(output, None, True)

        def quegan(image: ee.Image):
            """
            The following Multi-temporal speckle filters are implemented as described in
            S. quegan and J. J. Yu, “Filtering of multichannel SAR images,”
            IEEE Trans Geosci. Remote Sensing, vol. 39, Nov. 2001.
            this function will filter the collection used for the multi-temporal part
            it takes care of:
            - same image geometry (i.e relative orbit)
            - full overlap of image
            - amount of images taken for filtering
            -- all before
            -- if not enough, images taken after the image to filter are added
            Parameters:
            image : the Image to be filtered
            Returns:
            ee.Image: filtered image
            """

            def setresample(image: ee.Image):
                return image.resample()

            def get_filtered_collection(image: ee.Image):
                """
                Generate and returns a dedicated image collection
                Parameters:
                image : Image whose geometry is used to define the new collection
                """
                # filter collection over are and by relative orbit
                s1_coll = ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT') \
                    .filterBounds(image.geometry()) \
                    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
                    .filter(ee.Filter.listContains('transmitterReceiverPolarisation',
                                                   ee.List(image.get('transmitterReceiverPolarisation')).get(-1))) \
                    .filter(
                    ee.Filter.Or(ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_stop')),
                                 ee.Filter.eq('relativeOrbitNumber_stop', image.get('relativeOrbitNumber_start')))).map(
                    setresample)

                def check_overlap(_image: ee.Image):
                    """
                    A function that takes the image and checks for the overlap get all S1 frames from
                    this date intersecting with the image bounds
                    Parameters:
                    _image : Image to check the overlap with
                    Returns:
                    ee Image Collection: A collection with matching geometry
                    """

                    # get all S1 frames from this date intersecting with the image bounds
                    s1 = s1_coll.filterDate(_image.date(), _image.date().advance(1, 'day'))
                    # intersect those images with the image to filter
                    intersect = image.geometry().intersection(s1.geometry().dissolve(), 10)
                    # check if intersect is sufficient
                    valid_date = ee.Algorithms.If(intersect.area(10).divide(image.geometry().area(10)).gt(0.95),
                                                  _image.date().format('YYYY-MM-dd'))
                    return ee.Feature(None, {'date': valid_date})

                # this function will pick up the acq dates for fully overlapping acquisitions
                # before the image acquisition
                dates_before = s1_coll.filterDate('2014-01-01', image.date().advance(1, 'day')) \
                    .sort('system:time_start', False).limit(5 * self.no_of_images) \
                    .map(check_overlap).distinct('date').aggregate_array('date')

                # if the images before are not enough, we add images from after the image acquisition
                # this will only be the case at the beginning of S1 mission
                dates = ee.List(ee.Algorithms.If(dates_before.size().gte(self.no_of_images),
                                                 dates_before.slice(0, self.no_of_images),
                                                 s1_coll.filterDate(image.date(), '2100-01-01').sort(
                                                     'system:time_start', True)
                                                 .limit(5 * self.no_of_images).map(check_overlap).distinct('date')
                                                 .aggregate_array('date').cat(dates_before).distinct().sort().slice(0,
                                                                                                                    self.no_of_images)))

                # now we re-filter the collection to get the right acquisitions for multi-temporal filtering
                return ee.ImageCollection(
                    dates.map(lambda date: s1_coll.filterDate(date, ee.Date(date).advance(1, 'day'))
                              .toList(s1_coll.size())).flatten())

            # we get our dedicated image collection for that image
            s1 = get_filtered_collection(image)
            bands = image.bandNames().remove('angle')
            s1 = s1.select(bands)
            meanBands = bands.map(lambda bandName: ee.String(bandName).cat('_mean'))
            ratioBands = bands.map(lambda bandName: ee.String(bandName).cat('_ratio'))
            count_img = s1.reduce(ee.Reducer.count())

            def inner(image: ee.Image):
                """
                Creats an image whose bands are the filtered image and image ratio and
                returns filtered image and image ratio
                """
                _filtered = boxcar(image, self.kernel_size).select(bands).rename(meanBands)
                _ratio = image.select(bands).divide(_filtered).rename(ratioBands)
                return _filtered.addBands(_ratio)

            isum = s1.map(inner).select(ratioBands).reduce(ee.Reducer.sum())
            filtered = inner(image).select(meanBands)
            divide = filtered.divide(count_img)
            output = divide.multiply(isum).rename(bands)
            return image.addBands(output, None, True)

        return self.border_noise_correction().map(quegan)


    def slope_correction(self):
        """
        TERRAIN CORRECTION
        Parameters:
        collection : ee.ImageCollection
        terrain_flattening_model : string , the radiometric terrain normalization model, either volume or direct
        dem : ee asset, the dem to be used
        terrain_flattening_additioanl_layover_shadow_buffer : integer ,
            the additional buffer to account for the passive layover and shadow
        Returns
        ee image collection: An image collection where radiometric terrain normalization is implemented on each image
        """

        ninetyRad = ee.Image.constant(90).multiply(math.pi / 180)

        def _volumetric_model_SCF(theta_iRad: ee.Image, alpha_rRad: ee.Image):
            """
            Parameters:
            theta_iRad : the scene incidence angle
            alpha_rRad : the slope steepness in range
            Returns
            ee.Image: Applies the volume model in the radiometric terrain normalization
            """

            # Volume model
            nominator = (ninetyRad.subtract(theta_iRad).add(alpha_rRad)).tan()
            denominator = (ninetyRad.subtract(theta_iRad)).tan()
            return nominator.divide(denominator)

        def _direct_model_SCF(theta_iRad: ee.Image, alpha_rRad: ee.Image, alpha_azRad: ee.Image):
            """
            Parameters:
            theta_iRad : the scene incidence angle
            alpha_rRad : the slope steepness in range
            Returns
            ee.Image: applies the direct model in the radiometric terrain normalization
            """
            # Surface model
            nominator = (ninetyRad.subtract(theta_iRad)).cos()
            denominator = alpha_azRad.cos().multiply((ninetyRad.subtract(theta_iRad).add(alpha_rRad)).cos())
            return nominator.divide(denominator)

        def _erode(image: ee.Image, distance: int):
            """
            Parameters:
            image : ee.Image to apply the erode function to
            distance : integer, the distance to apply the buffer
            Returns:
            ee.Image : an image that is masked to compensate for passive layover
                and shadow depending on the given distance
            """
            d = (image.Not().unmask(1).fastDistanceTransform(30).sqrt()
                 .multiply(ee.Image.pixelArea().sqrt()))

            return image.updateMask(d.gt(distance))

        def _masking(alpha_rRad: ee.Image, theta_iRad: ee.Image, buffer: int):
            """
            Parameters:
            alpha_rRad :  Slope steepness in range
            theta_iRad : The scene incidence angle
            Returns:
            ee.Image:  An image that is masked to conpensate for passive layover
                and shadow depending on the given distance
            """
            # calculate masks
            # layover, where slope > radar viewing angle
            layover = alpha_rRad.lt(theta_iRad).rename('layover')
            # shadow
            shadow = alpha_rRad.gt(ee.Image.constant(-1)
                                   .multiply(ninetyRad.subtract(theta_iRad))).rename('shadow')
            # combine layover and shadow
            mask = layover.And(shadow)
            # add buffer to final mask
            if buffer > 0:
                mask = _erode(mask, buffer)
            return mask.rename('no_data_mask')

        def _correct(image: ee.Image):
            """
            Parameters:
            image : Image to apply the radiometric terrain normalization to
            Returns:
            ee.Image: Radiometrically terrain corrected image
            """

            bandNames = image.bandNames()
            geom = image.geometry()
            proj = image.select(1).projection()
            elevation = self.dem.resample('bilinear').reproject(crs=proj, scale=10).clip(geom)
            # calculate the look direction
            heading = ee.Terrain.aspect(image.select('angle')).reduceRegion(ee.Reducer.mean(), image.geometry(), 1000)
            # in case of null values for heading replace with 0
            heading = ee.Dictionary(heading).combine({'aspect': 0}, False).get('aspect')
            heading = ee.Algorithms.If(
                ee.Number(heading).gt(180),
                ee.Number(heading).subtract(360),
                ee.Number(heading)
            )
            # the numbering follows the article chapters
            # 2.1.1 Radar geometry
            theta_iRad = image.select('angle').multiply(math.pi / 180)
            phi_iRad = ee.Image.constant(heading).multiply(math.pi / 180)
            # 2.1.2 Terrain geometry
            alpha_sRad = ee.Terrain.slope(elevation).select('slope').multiply(math.pi / 180)
            aspect = ee.Terrain.aspect(elevation).select('aspect').clip(geom)
            aspect_minus = aspect.updateMask(aspect.gt(180)).subtract(360)
            phi_sRad = aspect \
                .updateMask(aspect.lte(180)) \
                .unmask() \
                .add(aspect_minus.unmask()) \
                .multiply(-1) \
                .multiply(math.pi / 180)
            self.dem.reproject(proj).clip(geom)
            # 2.1.3 Model geometry
            # reduce to 3 angle
            phi_rRad = phi_iRad.subtract(phi_sRad)
            # slope steepness in range (eq. 2)
            alpha_rRad = (alpha_sRad.tan().multiply(phi_rRad.cos())).atan()
            # slope steepness in azimuth (eq 3)
            alpha_azRad = (alpha_sRad.tan().multiply(phi_rRad.sin())).atan()
            # 2.2
            # Gamma_nought
            gamma0 = image.divide(theta_iRad.cos())
            if self.terrain_flattening_model == 'VOLUME':
                # Volumetric Model
                scf = _volumetric_model_SCF(theta_iRad, alpha_rRad)
            elif self.terrain_flattening_model == 'DIRECT':
                scf = _direct_model_SCF(theta_iRad, alpha_rRad, alpha_azRad)
            # apply model for Gamm0
            gamma0_flat = gamma0.multiply(scf)
            # get Layover/Shadow mask
            mask = _masking(alpha_rRad, theta_iRad, self.terrain_flattening_additional_layover_shadow_buffer)
            output = gamma0_flat.mask(mask).rename(bandNames).copyProperties(image)
            output = ee.Image(output).addBands(image.select('angle'), None, True)
            return output.set('system:time_start', image.get('system:time_start'))

        return self.multi_temporal_filter().map(_correct)
