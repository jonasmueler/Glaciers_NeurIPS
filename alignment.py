import numpy as np
import cv2
import os
#import torch
from config import *
import pickle
import glob
import matplotlib.pyplot as plt

def openData(name):
    """
    opens pickled data object
    
    name : string
    	named of saved data object
    	
    returns : list of tuple of datetime and np array 

    """
    with open(name, "rb") as fp:   # Unpickling
        data = pickle.load(fp)
    return data

def loadData(path, years, name):
    """

    path: string
        path to data pickle objects
    years: list of string
        years to be loaded
    returns: list of tuple of datetime and np.array
        date and image in list of tuple
    """

    os.chdir(os.path.join(path, "datasets", name, "rawData"))
    Path = os.getcwd()
    os.chdir(Path)
    print("Begin loading data")

    # read
    fullData = []
    for i in range(len(years)):
        helper = openData(years[i])
        fullData.append(helper)
    print("data loading finished")

    d = [item for sublist in fullData for item in sublist]

    return d

def kernel(x, mask):
    """
    x: np.array

    returns: float
        applied average kernel on one of the nan pixels
    """
    # Kernel from Maria-Minerva Vonica, Romania; Andrei Ancuta; Marc Frincu (2021)

    kernelMask = np.array([[1, 2, 3, 2, 1],
                           [2, 4, 6, 4, 2],
                           [3, 6, 9, 6, 3],
                           [2, 4, 6, 4, 2],
                           [1, 2, 3, 2, 1]])

    # get final kernel
    k = np.multiply(kernelMask, mask)

    # calculate weighted average
    res = np.ma.average(np.nan_to_num(np.ndarray.flatten(x)), weights=np.ndarray.flatten(k))

    return res

def applyToImage(img):
    """
    img: 2d np.array

    returns: 2d np.array

        array with imputed missings

    """

    # create matrix, where value is missing matrix is 0 otherwise 1
    missings = np.argwhere(np.isnan(img))
    zer = np.ones(img.shape)
    for i in range(len(missings)):
        zer[missings[i][0], missings[i][1]] = 0
        missings[i] = missings[i] + 2

    ## add 0 padding
    zer = np.vstack([np.zeros((2, len(zer[0, :]))), zer, np.zeros((2, len(zer[0, :])))])
    zer = np.hstack([np.zeros((len(zer[:, 0]), 2)), zer, np.zeros((len(zer[:, 0]), 2))])

    img = np.vstack([np.zeros((2, len(img[0, :]))), img, np.zeros((2, len(img[0, :])))])
    img = np.hstack([np.zeros((len(img[:, 0]), 2)), img, np.zeros((len(img[:, 0]), 2))])

    for i in range(len(missings)):
        # calculate value with kernel
        patch = img[missings[i][0] - 2:(missings[i][0] - 2) + 5, missings[i][1] - 2:(missings[i][1] - 2) + 5]
        mask = zer[missings[i][0] - 2:(missings[i][0] - 2) + 5, missings[i][1] - 2:(missings[i][1] - 2) + 5]
        res = kernel(patch, mask)
        img[missings[i][0], missings[i][1]] = res

    return img[2:-2, 2:-2]

def convertDatetoVector(date):
    """

    date: dateTime object
        time of picture taken
    returns: np.array
        vector of date
    """
    date = str(date)
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    res = np.array([day, month, year]) # applied changes here no more torch tensor !!

    return res


def enhancedCorAlign(imgStack, bands = [1,2,3]):
    """
    aligns images to mean temporal image

    imgStack: np.array
        (N, C, X, Y)
    bands: list
        bands to be aligned; default is RGB bands 
        For NDSI calculations use [2, 5, 5] (green and swir1 band), must be 
        three otherwise function has to be applied recursively multiple times
    returns: list of np.array
        [(C, X, Y)...]
    """

    # switch channel dimensions
    imgStack = np.transpose(imgStack, (0, 2, 3, 1))

    # create new array to fill
    imageStack = np.zeros((imgStack.shape[0], imgStack.shape[1], imgStack.shape[2], 3))

    # add 3rd channel dimension
    for i in range(len(imgStack)):
        imageStack[i] = np.stack((imgStack[i, :, :, bands[0]], imgStack[i, :, :, bands[1]], imgStack[i, :, :, bands[2]]), axis=2)

    # Compute average temporal image
    imgStack = imageStack.astype('float32')
    avgTemporalImage = np.mean(imgStack, axis=0).astype('float32')

    # median filter noise
    # Apply median filter to each channel separately
    filteredImg = np.zeros_like(avgTemporalImage)
    for i in range(2):
        filteredImg[:, :, i] = cv2.medianBlur(avgTemporalImage[:, :, i], 3)

    avgTemporalImage = filteredImg.astype('float32')

    # Compute gradient of averaged RGB channels
    grayAvgTemporalImage = cv2.cvtColor(avgTemporalImage, cv2.COLOR_BGR2GRAY).astype('float32')

    # Define motion model
    motionModel = cv2.MOTION_HOMOGRAPHY

    # Define ECC algorithm parameters
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 1e-4)

    #### debug ####
    #criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.1)
    ###############

    warpMatrix = np.eye(3, 3, dtype=np.float32)

    # Define ECC mask
    mask = None  # all pixels used as missings are already cleared

    # Apply ECC registration
    registeredStack = []
    counter = 0
    for frame in imageStack:
        grayFrame = cv2.cvtColor(frame.astype('float32'), cv2.COLOR_BGR2GRAY).astype('float32')
        (cc, warpMatrix) = cv2.findTransformECC(grayAvgTemporalImage, grayFrame, warpMatrix, motionModel,
                                                criteria, mask, 1)
        
        #registeredFrame = cv2.warpAffine(frame, warpMatrix, (frame.shape[1], frame.shape[0]),
        #                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        registeredFrame = cv2.warpPerspective(frame, warpMatrix, (frame.shape[1], frame.shape[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        registeredFrame = np.transpose(registeredFrame, (2, 0, 1))# [[0, 1], :, :] use if only two bands are needed for NDSI output
        registeredStack.append(registeredFrame)
        
        counter += 1

        print("scene: ", counter, "done")

    return registeredStack


def alignment(d, ROI, applyKernel, name):

    """
    gets list of list with images from  different months alignes the images

    d: list of tuple of datetime and np.array
    ROI: list of int
        region of interest to be processed in pixels
    applyKernel: boolean
        apply kernel to average out missing values in images
    name: str

    returns: tuple of np.array
        dates in np array and aligned images in np.array
    """

    # get region of interest in data
    for i in range(len(d)):
        img = d[i][1][:, ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # clean missings with kernel
        if applyKernel:
            # apply kernel to raw bands, do many times as sometimes division by zero gives nans in image
            for z in range(img.shape[0]):  
                while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                    img[z, :, :] = applyToImage(img[z, :, :])
                    print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
                print("band: ", z, " of ", img.shape[0] - 1, "done")
        print(f"application of kernel on image {i} done")
        d[i] = (d[i][0] ,img)

    # align data with enhanced correlation method
    print("start image alignment")
    images = [d[i][1] for i in range(len(d))]

    images = np.stack(images, axis = 0)

    alignedData = enhancedCorAlign(images)

    print("image alignment done")

    # put back into d list
    #for i in range(len(d)):
    #    d[i] = (d[i][0] , alignedData[i])
    
    # save on hard drive 
    os.chdir(path)
    os.makedirs(os.path.join(path, "datasets", name, "alignedData"), exist_ok=True)
    os.makedirs(os.path.join(path, "datasets", name, "dates"), exist_ok=True)

    # save data 
    for i in range(len(d)):
        # save images
        os.chdir(os.path.join(path, "datasets", name, "alignedData"))
        with open(str(i), "wb") as fp:
            pickle.dump(alignedData[i], fp)

        # save dates
        os.chdir(os.path.join(path, "datasets", name, "dates"))
        with open(str(i), "wb") as fp:
            pickle.dump(d[i][0], fp)

    return d

def minmaxScaler(X):
    """
    X: 2d array
    returns: 2d array
       values from [0,1]
    """
    #res = (X - np.nanpercentile(X,2)) / (np.nanpercentile(X, 98) - np.nanpercentile(X, 2))
    res = ((X - np.nanmin(X) )/ (np.nanmax(X) - np.nanmin(X))) * 255
    res = res.astype(np.uint8)
    
    return res

def createImage(img, alpha):
    """
    img: 3d array
        [red, green, blue]


    returns: np.array
        plot ready image
    """
    red = img[0, :,:]*alpha
    green = img[1, :,:]*alpha
    blue = img[2, :,:]*alpha

    green = minmaxScaler(green)
    blue = minmaxScaler(blue)
    red = minmaxScaler(red)

    plotData = np.dstack((red, green, blue))

    return plotData

def visualCheck(name):
    """
    plots the aligned and extracted images in RGB coordinates by using the scaled red, green and blue band values of the satellite.

    name: str
        name of the extracted data object from the satellite 
    """

    # load data
    currentPath = os.path.join(path, "datasets", name, "alignedData")
    os.chdir(currentPath)
    files = glob.glob(os.path.join(currentPath, '*'))

    # create folder 
    os.makedirs(os.path.join(path, "datasets", name, "alignedRGB"),exist_ok=True)

    # plot data sequentially and save
    for i in range(len(files)):  # rgb
        img = openData(files[i])
        img = createImage(img[:, :, :], 0.4) # alpha value hardcoded !!!
        plt.figure(figsize=(30,30))
        plt.imshow(img) #cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.savefig(os.path.join(path, "datasets", name, "alignedRGB", f"{i}.pdf"), dpi = 300, bbox_inches='tight')
        plt.clf()
    
    return None


"""


    # check for the correct months to make data stationary -> summer data
    l = []
    counterImg = 0
    usedMonths = []
    #months =  [[i, i+1, i+2] for i in range(1, 12, 3) if i+2 <= 12] # parvati
    months = [[i, i+1, i+2, i+3] for i in range(1, 13, 4) if i+3 <= 12]

    for y in np.arange(2013, 2022, 1): # year
        #for m in np.arange(1,13,1): # month
        for [m, n, t, h] in months:
            imgAcc = np.zeros((2, ROI[1] - ROI[0], ROI[3] - ROI[2]))
            month = 0
            for i in range(len(d)):
                if (((convertDatetoVector(d[i][0])[1].item() == m) or (convertDatetoVector(d[i][0])[1].item() == n)  or (convertDatetoVector(d[i][0])[1].item() == t) or (convertDatetoVector(d[i][0])[1].item() == h))  and (convertDatetoVector(d[i][0])[2].item() == y)):
                    # count months
                    month += 1

                    ## get roi and apply kernel
                    img = d[i][1][: , :, :]

                    # align
                    if month == 1:
                        imgAcc += img
                    if month > 1:
                        imgAcc = (imgAcc + img) / 2 # average


            # apply NDSI
            # add snow mask to average image
            threshold = 0.3
            NDSI = np.divide(np.subtract(imgAcc[0, :, :], imgAcc[1, :, :]),
                                         np.add(imgAcc[0, :, :], imgAcc[1, :, :]))
            # nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0) ## leave in case necessary to use in future
            snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
            l.append(snow)
            usedMonths.append(np.array([[m,y]]))

        print("averaging of year: ", y, " done")


    # sanity check
    #assert len(usedMonths) == len(l) == 9*3 # 9 years, 12 months, average 3 months


    # interpolate between images
    ## first images can not be interpolated take first not missing image as template and impute
    indices = [i for i in range(len(l)) if not np.any(np.isnan(l[i]))]

    for i in range(len(indices) - 1):
        idx = indices[i]
        succ = indices[i + 1]

        if succ - idx == 1:
            pass

        elif succ - idx > 1:
            diff = succ - idx
            delta = (l[succ] - l[idx]) / diff
            for t in range(diff):
                # l[idx + t + 1] = l[idx] + (t+1) * delta
                l[idx + t + 1] = np.ma.masked_where(l[idx] + (t + 1) * delta < threshold, NDSI).filled(0)

    print("interpolation done")
    result = l
    #usedMonths = usedMonths[1:]


    ## save on harddrive
    print("start saving scenes")
    path = os.getcwd()
    os.makedirs("monthlyAveragedScenes", exist_ok=True)
    os.chdir(os.path.join(path, "monthlyAveragedScenes"))
    os.makedirs("images", exist_ok=True)
    os.makedirs("dates", exist_ok=True)
    counter = 0

    for i in range(len(result)):
        # save images
        os.chdir(os.path.join(path,"monthlyAveragedScenes", "images"))

        # save data object on drive
        with open(str(counter), "wb") as fp:
            pickle.dump(result[i], fp)

        # save dates
        os.chdir(os.path.join(path, "monthlyAveragedScenes", "dates"))

        # save data object on drive
        with open(str(counter), "wb") as fp:
            pickle.dump(usedMonths[i], fp)
        counter += 1

    print("saving scenes done")

    return result

"""

def main(plot = True):
    os.chdir(path)
    d = loadData(path, years, name)
    d = alignment(d, extractedCoordinates, True, name)

    if plot:
        visualCheck(name)
    
    return None



if __name__ == "__main__":
    main()
     