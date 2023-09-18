import numpy as np
import cv2
import os
import torch
from config import *

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


def enhancedCorAlign(imgStack):
    """
    aligsn images to mean temporal image

    imgStack: np.array
        (N, C, X, Y)
    returns: list of np.array
        [(C, X, Y)...]
    """

    # switch channel dimensions
    imgStack = np.transpose(imgStack, (0, 2, 3, 1))

    # create new array to fill
    imageStack = np.zeros((imgStack.shape[0], imgStack.shape[1], imgStack.shape[2], 3))

    # add 3rd channel dimension
    for i in range(len(imgStack)):
        imageStack[i] = np.stack((imgStack[i, :, :, 0], imgStack[i, :, :, 1], imgStack[i, :, :, 1]), axis=2)

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
        registeredFrame = np.transpose(registeredFrame, (2, 0, 1))[[0, 1], :, :]
        registeredStack.append(registeredFrame)
        counter += 1

        print("scene: ", counter, "done")

    return registeredStack


def monthlyAverageScenesEnCC(d, ROI, applyKernel, name):

    """
    gets list of list with images from  different months alignes the images

    d: list of tuple of datetime and np.array
    ROI: list of int
        region of interest to be processed
    applyKernel: boolean
        apply kernel to average out missing values in images
    name: str

    returns: tuple of np.array
        dates in np array and aligned images in np.array
    """

    # get region of interest in data, relevant bands for NDSI index, clean missings
    for i in range(len(d)):
        img = d[i][1][[2,5], ROI[0]:ROI[1], ROI[2]:ROI[3]]
        # clean missings with kernel
        if applyKernel:
            # apply kernel to raw bands, do many times as sometimes division by zero gives nans in image
            for z in [0, 1]:  # only use NDSI relevant bands
                while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                    img[z, :, :] = applyToImage(img[z, :, :])
                    print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
                # img[z, :, :] = imageAlignArcosics(img[z, :, :], ref[z, :, :])
                print("band: ", z, " of ", img.shape[0], "done")
        print("application of kernel done")
        d[i] = (d[i][0] ,img)

    # align data with enhanced correlation method
    print("start image alignment")
    images = [d[i][1] for i in range(len(d))]

    images = np.stack(images, axis = 0)

    alignedData = enhancedCorAlign(images)

    print("image alignment done")

    # put back into d list
    for i in range(len(d)):
        d[i] = (d[i][0] , alignedData[i])
    
    # save on hard drive 
    os.chdir(path)
    os.makedirs(os.path.join(path, "datasets", name, "alignedData"))
    os.makedirs(os.path.join(path, "datasets", name, "Dates"))

    ## implement data save method here

    return d


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