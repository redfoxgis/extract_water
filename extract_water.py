#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 22:49:16 2019

@author: RedFox GIS & Remote Sensing, Aaron P.
"""
import os
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterstats import zonal_stats
from shapely.geometry import shape
from skimage import io
from skimage.segmentation import quickshift

INDATA = '/Users/aaron/gdrive/projects/segmentation_project/river_subset.tif'
OUTWS  = '/Users/aaron/gdrive/projects/segmentation_project/outdata'

def read_data(indata):
    dataset = rasterio.open(indata)
    
    # Get image bounding box info
    crs = dataset.crs
    affine = dataset.transform
    
    return dataset,crs,affine
    
def segment(indata):
    img = io.imread(indata)
    segments = quickshift(img, kernel_size=3, convert2lab=False, max_dist=6, ratio=0.5).astype('int32')
    print("Quickshift number of segments: %d" % len(np.unique(segments)))
    return segments

def ndvi_img(dataset):
    red = dataset.read(3).astype(float)
    nir = dataset.read(4).astype(float)

    np.seterr(divide='ignore', invalid='ignore') #Allow 0 in division

    ndvi = np.empty(dataset.shape, dtype=rasterio.float32)
    check = np.logical_or ( red > 0, nir > 0 )
    ndvi = np.where ( check,  (nir - red ) / ( nir + red ), -999 )
    return ndvi

def raster_to_poly(segments, affine):
    polys = []
    for shp, value in shapes(segments, transform=affine):
        polys.append(shp)
        
    # Convert json to geopandas df
    geom = [shape(i) for i in polys]
    polys_gdf = gpd.GeoDataFrame({'geometry':geom})
    return polys, polys_gdf

def zonal_statistics(polys, polys_gdf, ndvi, affine):
    # Calculate mean zonal stats of NDVI in segments
    zs = zonal_stats(polys, ndvi, affine = affine, stats="mean")
    # Add a ndvi field and calculate mean value
    se = pd.Series([f['mean'] for f in zs])
    polys_gdf['ndvi'] = se.values

    # Add a threshold field and populate with 1 if ndvi <-0.2
    def classifier(row):
        if row["ndvi"] < -0.2:
            return 1
        else:
            return 0
    
    polys_gdf["threshold"] = polys_gdf.apply(classifier, axis=1)

def export_to_shp(polys_gdf, dataset, outws):
    # Now export to shp to inspect results...
    polys_gdf.crs = dataset.crs
    polys_gdf.to_file(os.path.join(outws, "water.shp"))
    
def export_to_geotiff(polys_gdf, dataset, outws):
    # Create binary raster output of water features
    out_fn = os.path.join(outws, "water.tif")#'/Users/aaron/Desktop/temp/rasterizedFT.tif'

    meta = dataset.meta.copy()
    meta.update(compress='lzw')
    meta.update(count=1)

    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shps = ((geom,value) for geom, value in zip(polys_gdf.geometry, polys_gdf.threshold))

        burned = rasterio.features.rasterize(shapes=shps, fill=0, out=out_arr, transform=out.transform)
        out.write_band(1, burned)
        
def main():
    dataset,crs,affine = read_data(INDATA)
    segments = segment(INDATA)
    ndvi = ndvi_img(dataset)
    polys, polys_gdf = raster_to_poly(segments, affine)
    zonal_statistics(polys, polys_gdf, ndvi, affine)
#    export_to_shp(polys_gdf, dataset, OUTWS)
    export_to_geotiff(polys_gdf, dataset, OUTWS)

if __name__ == "__main__":
    main()
