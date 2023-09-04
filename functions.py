import numpy as np 
from numpy import array
import planetary_computer as pc
import pystac_client
import stackstac

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
        amount of clouds allowed in %
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
    print("found ", len(items), " scenes")

    # stack
    stack = stackstac.stack(items, bounds_latlon=bbox, epsg = "EPSG:32643")

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
                data = array([np.array(output[i, 0, :, :]),
                              np.array(output[i, 1, :, :]),
                              np.array(output[i, 2, :, :]),
                              np.array(output[i, 3, :, :]),
                              np.array(output[i, 4, :, :]),
                              np.array(output[i, 5, :, :]),
                              np.array(output[i, 6, :, :])
                              ])
                cloudCov = cloud[1][i]
                data = (time[i], data, cloudCov)
                dataList.append(data)

    return dataList