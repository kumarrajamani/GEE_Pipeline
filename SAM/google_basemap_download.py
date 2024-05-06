from qgis.core import *
from qgis.utils import *
from qgis.gui import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from qgis.core import QgsRasterFileWriter, QgsRectangle, QgsCoordinateReferenceSystem

# Define the bbox in EPSG:4326
# MH [74.0509619999999984,11.5823780000000003,78.5882950000000022,18.4767359999999989]
#[8.468486883,11.780617551,8.523076413,11.753748755]

#xmin, ymin, xmax, ymax = [77.4251581234569670,26.7411495864940711,78.8515725495735609,27.4088293750648795]
xmin, ymin, xmax, ymax = [8.468486883,11.780617551,8.523076413,11.753748755]
bbox_4326 = QgsRectangle(xmin, ymin, xmax, ymax)

# Define the CRS of the bbox in EPSG:4326
crs_4326 = QgsCoordinateReferenceSystem('EPSG:4326')

# Define the target CRS in EPSG:3857
crs_3857 = QgsCoordinateReferenceSystem('EPSG:3857')

# Define the coordinate transform
transform = QgsCoordinateTransform(crs_4326, crs_3857, QgsProject.instance())

# Transform the bbox to EPSG:3857
bbox_3857 = transform.transformBoundingBox(bbox_4326)

# Define the grid resolution in meters
grid_res = 500

# Define output resolution
resolution = 0.2

# Calculate the number of rows and columns in the grid
num_rows = int((bbox_3857.yMaximum() - bbox_3857.yMinimum()) / grid_res)
num_cols = int((bbox_3857.xMaximum() - bbox_3857.xMinimum()) / grid_res)

print(num_rows, num_cols)
layer = iface.activeLayer()  # Get the active layer
crs = layer.crs().authid()  # Get the layer CRS

for i in range(num_rows):
    for j in range(num_cols):
        # Calculate the bbox for the current grid cell
        grid_bbox_3857 = QgsRectangle(bbox_3857.xMinimum() + j * grid_res,
                                       bbox_3857.yMinimum() + i * grid_res,
                                       bbox_3857.xMinimum() + (j + 1) * grid_res,
                                       bbox_3857.yMinimum() + (i + 1) * grid_res)
        
        # Zoom to the bbox and 1:2000 scale
        layer.setExtent(grid_bbox_3857)
        iface.mapCanvas().setExtent(grid_bbox_3857)
        iface.mapCanvas().refresh()
        iface.mapCanvas().zoomScale(1000)
        
        # Calculate the width and height of the output file
        width = int(grid_bbox_3857.width() / resolution)
        height = int(grid_bbox_3857.height() / resolution)
        print(width, height)
        # Set up the output file path and name
        output_path = "C:/Users/raja.sivaranjan/Documents/SAM_tool/google_chips/africa"
        output_name = f"output_grid_{i}_{j}.tif"
        output = f"{output_path}/{output_name}"
        print(output)
        # Set up the QgsRasterPipe and write the GeoTIFF file
        renderer = layer.renderer()
        provider=layer.dataProvider()
        crs = layer.crs().toWkt()
        pipe = QgsRasterPipe()
        pipe.set(provider.clone())
        pipe.set(renderer.clone())
        file_writer = QgsRasterFileWriter(output)
        file_writer.writeRaster(pipe, width, height, grid_bbox_3857, layer.crs())

        # Add the GeoTIFF file as a new layer to the QGIS project
        # tif_layer = QgsRasterLayer(output, output_name, "gdal")
        # QgsProject.instance().addMapLayer(tif_layer)
