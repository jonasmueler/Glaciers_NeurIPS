# packages
import planetary_computer as pc
from numpy import array
import functions
import os
import pickle
from config import *

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
    d = functions.getData(bbox=box, bands=bands, timeRange=time, cloudCoverage= cloudCoverage, allowedMissings=allowedMissings)

    # save on hard drive with pickling
    # create folder
    pathOrigin = path + "/" + glacierName + "/" + "rawData"
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
                  glacierName):
    """
    years: list of string
        years that are to be extracted
    boundingBox: tuple of float
        tuple with four coordinates for the scene boundary
    clouds: int 
        int [0,100]
    allowedMissings: float
        [0,1], amount of missing pixels allowed
    glacierName: str
        Name of glacier and folder for files
    """
    for b in range(len(years) - 1):
        os.chdir(path)
        if b < 10:
            print("start processing year: " + years[b])
            string = years[b] + "-01-01/" + years[b+1] + "-01-01"
            API(boundingBox,
                string,
                clouds,
                allowedMissings, 
                years[b],
                glacierName)
            
            print(years[b] + " done")
    
    return None
        
     

if __name__ == "__main__":
    getYearlyData(years, boundingBox, clouds, allowedMissings, glacierName)














