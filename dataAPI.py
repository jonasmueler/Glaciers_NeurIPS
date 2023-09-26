# packages
import planetary_computer as pc
from numpy import array
import copy
from collections import Counter
import stackstac
import numpy as np
import pystac_client
import os
import pickle
from config import *

def getData(bbox, bands, timeRange, cloudCoverage, allowedMissings):
    """
    gets data in numpy format

    bbox: list of float
        rectangle to be printed
    bands: list of string
        ['coastal', 'blue', 'green', 'red', 'nir08', 'swir16', 'swir22', 'ST_QA',
       'lwir11', 'ST_DRAD', 'ST_EMIS', 'ST_EMSD', 'ST_TRAD', 'ST_URAD',
       'QA_PIXEL', 'ST_ATRAN', 'ST_CDIST', 'QA_RADSAT', 'SR_QA_AEROSOL']
    timeRange: string
        e.g. "2020-12-01/2020-12-31"
    cloudCoverage: int
        amount of clouds allowed in [0,100]
    allowedMissings: float
        amount of pixels nan

    returns: list of tuple of datetime array and 4d numpy array and cloudCoverage array
        [time, bands, x, y]

    """

    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

    # query
    search = catalog.search(
        collections=['landsat-8-c2-l2'],
        max_items= None,
        bbox=bbox,
        datetime=timeRange
    )

    items = pc.sign(search)
    print("found ", len(items), " images")

    # filter for different coordinate reference frames
    # get used epsgs
    epsgs = [items.items[i].properties["proj:epsg"] for i  in range(len(items.items))]
    
    # filter for different epsgs
    unique = Counter(epsgs)
    print(f"Found {len(unique.keys())} different epsg signatures in items: {unique}")
    print("Items with different epsgs are now processed seperately and then merged. This is okay as the images are aligned in a later step anyways")

    # loop over different epsgs
    completeData = []
    for i in range(len(unique.keys())):
        epsg = list(unique.keys())[i]
        print(f"Processing epsg: {epsg}")
        epsgInd = list(map(lambda x: True if x == epsg else False, epsgs))
        itemsInd = copy.deepcopy(items)
        itemsInd.items = [i for (i, v) in zip(itemsInd.items, epsgInd) if v]

        # stack
        stack = stackstac.stack(itemsInd, bounds_latlon=bbox) #, epsg = "EPSG:32644")

        # use common_name for bands
        stack = stack.assign_coords(band=stack.common_name.fillna(stack.band).rename("band"))
        output = stack.sel(band=bands)

        # put into dataStructure
        t = output.shape[0]
        cloud = np.array(output["eo:cloud_cover"] <= cloudCoverage)
        cloud = [cloud, np.array(output["eo:cloud_cover"])]
        time = np.array(output["time"])

        dataList = []
        for i in range(t):
            if cloud[0][i] == True:  # check for clouds
                if np.count_nonzero(np.isnan(output[i, 1, :, :])) >= round(
                        (output.shape[2] * output.shape[3]) * allowedMissings):  # check for many nans
                    pass
                elif np.count_nonzero(np.isnan(output[i, 1, :, :])) <= round(
                        (output.shape[2] * output.shape[3]) * allowedMissings):
                    data = array([np.array(output[i, 0, :, :]), # get seven bands of the satellite sensors
                                np.array(output[i, 1, :, :]),
                                np.array(output[i, 2, :, :]),
                                np.array(output[i, 3, :, :]),
                                np.array(output[i, 4, :, :]),
                                np.array(output[i, 5, :, :]),
                                np.array(output[i, 6, :, :])
                                ])
                    cloudCov = cloud[1][i]
                    print(time[i])
                    data = (time[i], data, cloudCov)
                    dataList.append(data)
        print("done!")
        completeData += dataList

    return completeData

# wrapper for acquiring data
def API(box, 
        time, 
        cloudCoverage, 
        allowedMissings, 
        year, 
        glacierName, 
        bands = ['coastal', 'red', 'green', 'blue', 'nir08', 'swir16', 'swir22']):
    """
    acquire and preprocess the data

    box: tuple of float
        coordinate box from which images are taken
    time: string
        time range for extraction of data
    cloudCoverage: int
        percent of pixels covered with clouds in bands
    allowedMissings: float
        p(missingData)
    year: string
        year of data extraction, downloads chunks of data as one year packages
    glacierName: string
        Name of the glacier for folder structure

    return: list of tuple of datetime and 4d ndarray tensor for model building
    """
    # get data
    d = getData(bbox=box, bands=bands, timeRange=time, cloudCoverage= cloudCoverage, allowedMissings=allowedMissings)

    # save on hard drive with pickling
    # create folder
    pathOrigin = path + "/" + "datasets" + "/" + glacierName + "/" + "rawData"
    os.makedirs(pathOrigin, exist_ok = True)
    os.chdir(pathOrigin)

    # save data object
    with open(year, "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")
    return d

# get Data and save seperately for each year in a file
def getYearlyData(years, 
                  boundingBox, 
                  clouds, 
                  allowedMissings, 
                  name):
    """
    years: list of string
        years that are to be extracted
    boundingBox: tuple of float
        tuple with four coordinates for the scene boundary
    clouds: int 
        int [0,100]
    allowedMissings: float
        [0,1], amount of missing pixels allowed
    Name: str
        Name of folder
    """

    for b in range(len(years)): #- 1):
        os.chdir(path)
        if b < 10: ## fix here !!
            print("#####################################################################################################")
            print("start processing year: " + years[b])
            #string = years[b] + "-01-01/" + years[b+1] + "-01-01"
            string = years[b] + "-01-01/" + str(int(years[b]) + 1) + "-01-01"
            
            API(boundingBox,
                string,
                clouds,
                allowedMissings, 
                years[b],
                name)
            
            print(years[b] + " done")
    
    return None
        
     

if __name__ == "__main__":
    getYearlyData(years, boundingBox, clouds, allowedMissings, name)














