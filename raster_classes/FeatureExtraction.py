import geopandas as gpd
import ee
import logging
import pandas as pd
import time
from functools import reduce
import requests

class FeatureExtraction:
    def __init__(self, fc, rc_list, file_name, logger, sampleRegions):
        self.fc = fc
        self.rc_list = rc_list
        self.file_name = file_name
        self.logger = logger
        self.sampleRegions = sampleRegions

    def getRequests(self):
        df_li = []
        for i in range(0, len(self.rc_list), 2):
            _st_time = time.time()
            img = ee.Image([self.rc_list[i].set_aoi(self.fc).image,
                            self.rc_list[i + 1].set_aoi(self.fc).image])
            if self.sampleRegions:
                feature_sample = img.sampleRegions(collection=self.fc,
                                                   scale=10,
                                                   projection='EPSG:4326',
                                                   geometries=True)
            else:
                feature_sample = img.reduceRegions(**{'collection': self.fc,
                                                      'reducer': ee.Reducer.median(),
                                                      'scale': 10,
                                                      'crs': 'EPSG:4326'})

            fname = self.file_name + '_t' + str(int(i / 2))

            url = feature_sample.getDownloadURL(filetype='geojson', selectors=None, filename=fname)
            # url_li.append(url)
            try:
                gdf = gpd.read_file(url)
            except Exception as e:
                if '/vsimem' in str(e):
                    self.logger.info(f"Skipping file {url} due to unsupported file format")
                    continue
            try:
                if isinstance(gdf, gpd.GeoDataFrame):
                    df_li.append(pd.DataFrame(gdf))
            except Exception as ex:
                if "'gdf' referenced before assignment" in str(ex):
                    gdf = gpd.read_file(url)
                    df_li.append(pd.DataFrame(gdf))
                else:
                    self.logger.info(f"Skipping file {fname} with {url} due to {ex}")
                    continue
            el_time = time.time() - _st_time
            self.logger.info("Processing time for {} is {} min".format(fname, (el_time / 60)))
        return df_li

    def merge_df(self, df_li):
        # df_li = []
        # for url in url_li:
        #     try:
        #         gdf = gpd.read_file(url)
        #     except Exception as e:
        #         if '/vsimem' in str(e):
        #             self.logger.info(f"Skipping file {url} due to unsupported file format")
        #             continue
        #         else:
        #             self.logger.info(f"Error reading file {url}: {e}")
        #             break
        #     df_li.append(pd.DataFrame(gdf))
        sh_li = []
        for df in df_li:
            sh_li.append(df.shape[0])
        max_id = sh_li.index(max(sh_li))
        final_df = df_li[max_id]

        for li_id in range(0, len(df_li)):
            if li_id == max_id:
                continue
            else:
                final_df = pd.merge(final_df, df_li[li_id], how="outer", on=["id"], suffixes=('', '_y'))
                final_df.drop(final_df.filter(regex='_y$').columns, axis=1, inplace=True)

        return final_df

    def execute(self):
        url_li = self.getRequests()
        final_df = self.merge_df(url_li)
        return final_df




