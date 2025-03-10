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
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, maxIterations, stoppingCriterion)

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
    gets list of list with images from  different months aligns the images

    d: list of tuple of datetime and np.array
    ROI: list of int
        region of interest to be processed in pixels
    applyKernel: boolean
        apply kernel to average out missing values in images
    name: str

    returns: None
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

    return None


def alignmentNDSIBands(d, ROI, applyKernel, name):

    """
    gets list of list with images from  different months, aligns the images

    d: list of tuple of datetime and np.array
    ROI: list of int
        region of interest to be processed in pixels
    applyKernel: boolean
        apply kernel to average out missing values in images
    name: str

    returns: None
        
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

    alignedData = enhancedCorAlign(images, bands = [2, 5, 5])

    print("image alignment done")

    # put back into d list
    #for i in range(len(d)):
    #    d[i] = (d[i][0] , alignedData[i])
    
    # save on hard drive 
    os.chdir(path)
    os.makedirs(os.path.join(path, "datasets", name, "alignedDataNDSIBands"), exist_ok=True)
    os.makedirs(os.path.join(path, "datasets", name, "dates"), exist_ok=True)

    # save data 
    for i in range(len(d)):
        # save images
        os.chdir(os.path.join(path, "datasets", name, "alignedDataNDSIBands"))
        with open(str(i), "wb") as fp:
            pickle.dump(alignedData[i], fp)
        
        # save dates
        os.chdir(os.path.join(path, "datasets", name, "dates"))
        with open(str(i), "wb") as fp:
            pickle.dump(d[i][0], fp)

    return None

def calculateNDSI():
    """
    calculates NDSI images in new folder from extracted data

    """

    # load data
    currentPath = os.path.join(path, "datasets", name, "alignedDataNDSIBands")
    os.chdir(currentPath)
    files = glob.glob(os.path.join(currentPath, '*'))

    # create folder for NDSI images 
    os.makedirs(os.path.join(path, "datasets", name, "alignedDataNDSI"), exist_ok=True)

    # load and change data 
    for i in range(len(files)):
        img = openData(files[i])
        threshold = 0.3
        NDSI = np.divide(np.subtract(img[0, :, :], img[1, :, :]),
                                         np.add(img[0, :, :], img[1, :, :]))
        
        # nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0) ## leave in case necessary to use in future
        snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
        
        # save 
        os.chdir(os.path.join(path, "datasets", name, "alignedDataNDSI"))
        with open(str(i), "wb") as fp:
            pickle.dump(snow, fp)

    return

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

def visualCheck(name, aligned = False, alignedAveraged = True):
    """
    plots the aligned and extracted images in RGB coordinates by using the scaled red, green and blue band values of the satellite.

    name: str
        name of the extracted data object from the satellite 
    aligned: boolean
        plot aligned images?
    alignedAveraged: boolean 
        plot aligned and averaged images?
    """
    if aligned: 
        # load data
        if extractNDSI == True:
            currentPath = os.path.join(path, "datasets", name, "alignedDataNDSI")
        elif extractNDSI == False:
            currentPath = os.path.join(path, "datasets", name, "alignedData")
        os.chdir(currentPath)
        files = glob.glob(os.path.join(currentPath, '*'))

        # create folder 
        os.makedirs(os.path.join(path, "datasets", name, "alignedPlots"),exist_ok=True)

        # plot data sequentially and save
        for i in range(len(files)):  # rgb
            img = openData(files[i])
            if extractNDSI:
                pass
            elif extractNDSI == False: #create normalized RGB image
                img = createImage(img[:, :, :], 0.4) # alpha value hardcoded !!!
            plt.figure(figsize=(30,30))
            plt.imshow(img) #cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.axis('off')
            plt.savefig(os.path.join(path, "datasets", name, "alignedPlots", f"{i}.pdf"), dpi = 300, bbox_inches='tight')
            plt.clf()
    elif alignedAveraged:
        # load data
        if extractNDSI == True:
            currentPath = os.path.join(path, "datasets", name, "alignedAveragedDataNDSI")
        elif extractNDSI == False:
            currentPath = os.path.join(path, "datasets", name, "alignedAveragedData")
        os.chdir(currentPath)
        files = glob.glob(os.path.join(currentPath, '*'))

        # create folder 
        os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedPlots"),exist_ok=True)

        # plot data sequentially and save
        for i in range(len(files)):  # rgb
            img = openData(files[i])
            if extractNDSI:
                plt.figure(figsize=(30,30))
                plt.imshow(img)
            elif extractNDSI == False: #create normalized RGB image
                img = createImage(img[:, :, :], 0.4) # alpha value hardcoded !!!
                plt.figure(figsize=(30,30))
                plt.imshow(img) #cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            #plt.axis('off')
            plt.savefig(os.path.join(path, "datasets", name, "alignedAveragedPlots", f"{i}.pdf"), dpi = 300, bbox_inches='tight')
            plt.clf()

    
    return None

def averageOverMonths(delta = 6, verbose = True):
    """
    Takes aligned images and averages over specified timedelta

    delta: int 
        time delta to average over 
    
    verbose: boolean 
        print information about image distribution for mean estimates

    """
    print("######################################################################################################")
    print("start averaging over months")
    os.chdir(os.path.join(path, "datasets", name, "dates"))
    filesGlob = glob.glob(os.path.join(os.path.join(path, "datasets", name, "dates"), '*'))

    output = []
    for year in years:
        print("processing: " + year)
        # filter for years
        files = [filesGlob[i] for i in range(len(filesGlob)) if int(convertDatetoVector(openData(filesGlob[i]))[2]) == int(year)]

        # filter and average over months in year
        listOfList = []
        listOfListInd = []
        for i in range(1, 12, delta): # deltas
            monthsDelta = [convertDatetoVector(openData(files[b]))[1] for b in range(len(files)) if 
                        convertDatetoVector(openData(files[b]))[1] < i + (delta) and convertDatetoVector(openData(files[b]))[1] >= i ]
            monthsDeltaInd = [b for b in range(len(files)) if 
                        convertDatetoVector(openData(files[b]))[1] < i + (delta) and convertDatetoVector(openData(files[b]))[1] >= i ]
            listOfList.append(monthsDelta)
            listOfListInd.append(monthsDeltaInd)
        
        if verbose:
            print(f"Distribution of images in timedeltas: {listOfListInd}")

        # check if no images for at least one delta
        if any(not sublist for sublist in listOfList):
            print("At least one timedelta interval contains no images")
            print("Choose different time-delta or different coordinates")
            return None
        
        # average data arrays over months
        #imagesYear = []
        for i in range(len(listOfListInd)):
            if extractNDSI == True: 
                images = [openData(os.path.join(path, "datasets", name, "alignedDataNDSI", str(ind))) for ind in listOfListInd[i]]
            if extractNDSI == False: 
                images = [openData(os.path.join(path, "datasets", name, "alignedData", str(ind))) for ind in listOfListInd[i]]
            imagesAvg = np.mean(np.stack(images, 0), 0)
            output.append(imagesAvg)
        
        #output.append(imagesYear)

    # save averaged images on harddrive
    for i in range(len(output)):
        if extractNDSI == True: 
            os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedDataNDSI"), exist_ok=True)
            os.chdir(os.path.join(path, "datasets", name, "alignedAveragedDataNDSI"))
            with open(str(i), "wb") as fp:
                pickle.dump(output[i], fp)
                
        if extractNDSI == False: 
            os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedData"), exist_ok=True)
            os.chdir(os.path.join(path, "datasets", name, "alignedAveragedData"))
            with open(str(i), "wb") as fp:
                pickle.dump(output[i], fp)
    
    return output

def main(plot = True):
    os.chdir(path)
    d = loadData(path, years, name)
    if extractNDSI == True:
        d = alignmentNDSIBands(d, extractedCoordinates, True, name)
        d = calculateNDSI()
        d = averageOverMonths(delta = deltaT)
        if plot:
            visualCheck(name)

    elif extractNDSI == False:
        d = alignment(d, extractedCoordinates, True, name)
        d = averageOverMonths(delta = deltaT)
        if plot:
            visualCheck(name)
    
    return None

if __name__ == "__main__":
    main()
    
     