# packages
import coiled
import distributed
import dask
import pandas as pd
import pystac_client
import planetary_computer as pc
import ipyleaflet
import IPython.display as dsp
import geogif
from dateutil.parser import ParserError
import stackstac
import bottleneck
import dask
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from numpy import array
import cv2
import imutils
from torch import nn
from numpy import linalg as LA
from numpy import ma
import os
import pickle
from sklearn.feature_extraction import image
import torch.optim as optim
import torch
# memory overflow bug fix
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
#
#import sys
#from torchvision import transforms
from PIL import Image
import wandb
from torch.autograd import Variable
from collections import Counter
from geoarray import GeoArray
from arosics import COREG, COREG_LOCAL



## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"



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

# visualize RGB images
# convert to RGB scale

def minmaxConvertToRGB(X, ret=False):
    """
    X: 2d array
    returns: 2d array
       values from 0-255
    """
    if ret == False:
        res = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
        res = res * 255.999
        return res
    elif ret:
        res = (X - np.nanmin(X)) / (np.nanmax(X) - np.nanmin(X))
        res = res * 255.999
        return [res, np.nanmin(X), np.nanmax(X)]

def minmaxScaler(X):
    """
    X: 2d array
    returns: 2d array
       values from [0,1]
    """
    #res = (X - np.nanpercentile(X,2)) / (np.nanpercentile(X, 98) - np.nanpercentile(X, 2))
    res = ((X - np.nanmin(X) )/ (np.nanmax(X) - np.nanmin(X))) #*255.99
    #res = res.astype("uint8")
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

# kernel
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


# create mean image (unaligned) and use it for the imputation of remainig missnig values in the satellite image
def imputeMeanValues(d, band):
    """
    creates mean image nad imputes values for areas which are not covered from the satelite

    d: list of tuple of datetime and ndarray
    bands: bands to be averaged over

    returns: list of tuple of datetime and ndarray
        with imputed values over the edges
    """
    # get images without missing corners
    idx = []
    idxMissing = []
    for i in range(len(d)):
        if np.sum(np.isnan(d[i][1][band, :, :])) == 0:
            idx.append(i)
        if np.sum(np.isnan(d[i][1][band, :, :])) > 0:
            idxMissing.append(i)

    Mean = d[idx[0]][1][band, :, :]
    for i in idx[1:]:
        Mean += d[i][1][band, :, :]

    Mean = Mean / len(idx)
    # impute mean values into images with missing corners
    for z in range(len(idxMissing)):
        img = d[idxMissing[z]][1][band, :, :]
        missings = np.argwhere(np.isnan(img))

        for x in range(len(missings)):
            insert = Mean[missings[x][0], missings[x][1]]
            img[missings[x][0], missings[x][1]] = insert
        d[idxMissing[z]][1][band, :, :] = img

    return d


# calculate NDSI values; bands not correct, check !! adapt if all eleven bands are used
def NDSI(Input, threshold):
    """
    creates three new images: NDSI, snow-mask, no-snow-mask

    Input: list of tuple of datetime and 3d ndarray
    threshold: float
        threshold for NDSI masks ~ 0.3-0.6 usually

    returns: list of tuple of datetime and 3d ndarray
        switch swir Band with calculated NDSI values
    """
    """
    for i in range(len(Input)):
        #tensor = Input[i][1][:, :, :]
        tensor = Input
        NDSI = np.divide(np.subtract(tensor[2, :, :], tensor[5, :, :]), np.add(tensor[2, :, :], tensor[5, :, :]))
        nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0)
        snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
        switchD = np.dstack((tensor[0, :, :],tensor[1, :, :],tensor[2, :, :],tensor[3, :, :],tensor[4, :, :],
                             tensor[5, :, :],tensor[6, :, :], NDSI, nosnow, snow))
        switchD = np.transpose(switchD, (2,0,1)) # switch dimensions back
        Input[i] = (Input[i][0], switchD)
    """
    tensor = Input
    NDSI = np.divide(np.subtract(tensor[2, :, :], tensor[5, :, :]), np.add(tensor[2, :, :], tensor[5, :, :]))
    nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0)
    snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)
    switchD = np.dstack((tensor[0, :, :], tensor[1, :, :], tensor[2, :, :], tensor[3, :, :], tensor[4, :, :],
                         tensor[5, :, :], tensor[6, :, :], NDSI, nosnow, snow))
    switchD = np.transpose(switchD, (2, 0, 1))  # switch dimensions back

    return switchD

# blur imputed pixels to cover edge of imputation
def gaussianBlurring(Input, kSize, band):
    """
    Input: list of tuple of datetime and 3d ndarray

    returns: list of tuple of datetime and 3d ndarray
        with applied filter on band 1 and 3
    """
    for i in range(len(Input)):
        Input[i][1][band, :, :] = cv2.GaussianBlur(Input[i][1][band, :, :], (kSize, kSize), 0)
    return Input


### image alignment with ORB features and RANSAC algorithm 
def alignImages(image, template, RGB, maxFeatures, keepPercent):
    """
    image: 2d or 3d nd array
        input image to be aligned
    template: 2d or 3d nd array
        template for alignment
    RGB: boolean
        is image in RGB format 3d ndarray?
    maxFeatures: int
        max. amount of features used for alignment
    keepPercent: float
        amount of features kept for aligning



    returns: ndarray
        alignend image

    """

    # convert both the input image and template to grayscale
    if RGB:
        imageGray = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2GRAY)
        templateGray =  cv2.cvtColor(template.astype('uint8'), cv2.COLOR_BGR2GRAY)
    if RGB == False:
        imageGray = image
        imageGray = imageGray.astype('uint8')

        templateGray = template
        templateGray = templateGray.astype('uint8')

    # use ORB to detect keypoints and extract features
    orb = cv2.ORB_create(maxFeatures)
    (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
    (kpsB, descsB) = orb.detectAndCompute(templateGray, None)


    # match the features
    #method = cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING
    #matcher = cv2.DescriptorMatcher_create(method)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descsA, descsB) #, None)

    # sort 
    matches = sorted(matches, key=lambda x:x.distance)

    # keep only the top 
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]


    # allocate memory for the keypoints (x, y)-coordinates from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # loop over the top matches
    for (i, m) in enumerate(matches):
        # indicate that the two keypoints in the respective images
        # map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # compute the homography matrix between the two sets of matched
    # points
    #print(ptsA)
    #print(ptsB)
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # use the homography matrix to align the images
    (h, w) = template.shape[:2]

    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned

def aligneOverTime(data):
    """
    alligns 6 consecutive scenes a extracted mean image from the 6 scenes

    data: list of torch tensor

    returns: list of torch.tensor

        list of aligned images

    """

    flows = []
    for i in range(len(data) - 1):
        img = data[i + 1]
        reference = data[i]
        helper = alignImages(img, reference, 5000, 0.25)
        flow = opticalFlow(img, reference, False)
        data[i+1] = helper
        flows.append(flow)

    # save images on hard drive
    #os.makedirs("alignedPatches")
    #os.chdir(os.getcwd(), "alignedPatches")
    #for i in range(len(data)):
    #    with open(str(i), "wb") as fp:  # Pickling
    #        pickle.dump(data[i], fp)

    return data

def opticalFlow(frame1, frame2, plot):
    """
    calculates optical flow between images

    frame1: np.array
    frame2: np.array
    plot: boolean
    return: float
        avg visual flow between images
    """
    frame1 = frame1.astype(np.float32)
    frame2 = frame2.astype(np.float32)
    frame1= cv2.merge([frame1,frame1, frame1])
    frame2 = cv2.merge([frame2, frame2, frame2])


    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Compute optical flow using Lucas-Kanade method
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 6, 13, 10, 5, 1.1, 0)

    # Visualize optical flow
    if plot:
        h, w = prev_gray.shape
        x, y = np.meshgrid(np.arange(0, w, 5), np.arange(0, h, 5))
        x_flow = flow[..., 0][::5, ::5]
        y_flow = flow[..., 1][::5, ::5]
        plt.quiver(x, y, x_flow, y_flow)



    # Compute magnitude and angle of optical flow
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Visualize magnitude of optical flow
    #plt.imshow(mag, cmap='gray')
    #plt.show()

    # Compute average magnitude of optical flow as a measure of glacier motion
    avg_mag = np.mean(mag)
    print("Average magnitude of optical flow:", avg_mag)

    return avg_mag

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

def loadData(path, years):
    """

    path: string
        path to data pickle objects
    years: list of string
        years to be loaded
    returns: list of tuple of datetime and np.array
        date and image in list of tuple
    """

    os.chdir(path)
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

## create time series of 5 images
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
    res = torch.tensor([day, month, year], dtype = torch.float)

    return res

def saveCheckpoint(model, optimizer, filename):
    """
    saves current model and optimizer step

    model: nn.model
    optimizer: torch.optim.optimzer class
    filename: string
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, os.path.join(pathOrigin, filename))
    print("checkpoint saved")
    return
def loadCheckpoint(model, optimizer, path):
    """
    loads mode and optimzer for further training
    model: nn.model
    optimizer: torch.optim.optimzer class
    path: string 
    return: list of optimizer and model
     
    """
    if optimizer != None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("checkpoint loaded")
        return [model, optimizer]
    elif optimizer == None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        return model

def tokenizerBatch(model, x, mode, device, flatten = torch.nn.Flatten(start_dim=1, end_dim=2)):
    """
    encoding and decoding function for transformer model

    model: nn.Module
    x: tensor
    mode: string
    device: string
    flatten: nn.Flatten for latent space input
    return: torch.tensor
        encoding/decoding for tarnsformer model
    """
    model.eval()
    if mode == "encoding":
        encoding = [model.encoder(x[i, :, :, :].to(device))[0] for i in range(x.size(0))]
        encoding = torch.stack(encoding).squeeze(dim = 2)

        return encoding

    if mode == "decoding":
        decoding = [model.decoder(x[i, :, :].to(device)) for i in range(x.size(0))]
        decoding = torch.stack(decoding)

        return decoding


def trainLoop(trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin):
    """
    trains a given model on the data

    dataLoader: torch DataLoader object
    valLoader: torch DataLoader object
    tokenizer: nn.Module
        trained tokenizer, fixed in training
    model: torch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive with path
    params: dict
        lr, weightDecay, epochs, batchSize, validationStep, optimizer
    WandB: boolean
        use weights and biases tool to monitor losses dynmaically
    device: string
        device on which the data should be stored

    return: nn.class
        trained model and saved training monitoring data
    """
    # variables
    torch.autograd.set_detect_anomaly(True)
    trainLosses = np.ones(len(trainLoader) * params["batchSize"])
    validationLosses = np.ones(len(trainLoader) * params["batchSize"])
    epochs = np.ones(len(trainLoader) * params["batchSize"])
    trainCounter = 0
    valLoss = torch.zeros(1)
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config={
                "learning_rate": params["learningRate"],
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": params["epochs"],
            }
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/models")
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/models/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    ###################### start training #############################


    for b in range(params["epochs"]):
        for inpts, targets in trainLoader:
            model.train()
            # use tokenizer on gpu
            inpts = inpts.to(device).float() # three maps, just use snow map as input
            targets = targets.to(device).float()

            if tokenizer != None:
                #targetsOrg = targets.clone()
                # encode with tokenizer and put to gpu
                inpts = tokenizerBatch(tokenizer, inpts, "encoding", device)
                targets = tokenizerBatch(tokenizer, targets, "encoding", device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                forward = model.forward(inpts, targets, training = True)
                #forward = tokenizerBatch(tokenizer, forward, "decoding", device)
                #forward = torch.reshape(forward, (forward.size(0), forward.size(1),  50, 50))
                loss = criterion(forward, targets)

            if tokenizer == None:
               # targetsOrg = targets.clone()

                # encode with tokenizer and put to gpu
                inpts = torch.flatten(inpts, start_dim = 2, end_dim = 3)
                targets = torch.flatten(targets, start_dim = 2, end_dim = 3)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                forward = model.forward(inpts, targets, training=True)

               # forward = torch.reshape(forward, (forward.size(0), forward.size(1), 50, 50))
                loss = criterion(forward[0], targets) + forward[1]


            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0) # gradient clipping; no exploding gradient
            optimizer.step()


            # save loss
            with torch.no_grad():
                if trainCounter % params["validationStep"] == 0 and trainCounter != 0:
                    model.eval()
                    x, y = next(iter(valLoader))
                    x = x.float().to(device)
                    y = y.float().to(device)
                    #yOrg = y.float().clone().to(device)

                    if tokenizer != None:
                        # encode with tokenizer and put to gpu
                        x = tokenizerBatch(tokenizer, x, "encoding", device)
                        y = tokenizerBatch(tokenizer, y, "encoding", device)

                        # predict
                        pred = model.forward(x,y, training = False)
                        #pred = tokenizerBatch(tokenizer, pred, "decoding", device)
                        #pred = torch.reshape(pred, (pred.size(0), pred.size(1),  50, 50))
                        valLoss = criterion(pred, y) 

                    if tokenizer == None:
                        # encode with tokenizer and put to gpu
                        x = torch.flatten(x, start_dim = 2, end_dim = 3)
                        y = torch.flatten(y, start_dim = 2, end_dim = 3)

                        # predict
                        pred = model.forward(x, y, training=False)
                       # pred = torch.reshape(pred, (pred.size(0), pred.size(1), 50, 50))
                        valLoss = criterion(pred[0], y) + pred[1]


                ## log to wandb
                if WandB:
                    wandb.log({"trainLoss": loss.detach().cpu().item(),
                            "validationLoss": valLoss.detach().cpu().item()})

                #save for csv
                trainLosses[trainCounter] = loss.detach().cpu().item()
                validationLosses[trainCounter] = valLoss.detach().cpu().item()
                epochs[trainCounter] = b
                trainCounter += 1

            # save model and optimizer checkpoint in case of memory overlow
            if trainCounter % 500 == 0:
                saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)

                # save gradient descent
                df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
                df.to_csv(os.path.join(pathOrigin, modelName) + ".csv", index=False)

            # print loss
            print("epoch: ", b, ", example: ", trainCounter, " current loss = ", loss.detach().cpu().item())


    # save results of gradient descent
    path = os.path.join(pathOrigin, "/models")
    os.chdir(path)
    df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
    df.to_csv(modelName + ".csv", index=False)

    ## save model state
    saveCheckpoint(model, optimizer, modelName)
    print("results saved!")
    return

def trainLoopClassification(trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin):
    """
    trains a given model on the data

    dataLoader: torch DataLoader object
    valLoader: torch DataLoader object
    tokenizer: nn.Module
        trained tokenizer, fixed in training
    model: torch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive with path
    params: dict
        lr, weightDecay, epochs, batchSize, validationStep, optimizer
    WandB: boolean
        use weights and biases tool to monitor losses dynmaically
    device: string
        device on which the data should be stored

    return: nn.class
        trained model and saved training monitoring data
    """
    # variables
    torch.autograd.set_detect_anomaly(True)
    trainLosses = np.ones(len(trainLoader) * params["batchSize"])
    validationLosses = np.ones(len(trainLoader) * params["batchSize"])
    epochs = np.ones(len(trainLoader) * params["batchSize"])
    trainCounter = 0
    valLoss = torch.zeros(1)
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config={
                "learning_rate": params["learningRate"],
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": params["epochs"],
            }
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/models")
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/models/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    ###################### start training #############################


    for b in range(params["epochs"]):
        for inpts, targets in trainLoader:
            model.train()
            # use tokenizer on gpu
            inpts = inpts.to(device).float() # three maps, just use snow map as input
            targets = targets.to(device).float()
            targets = (targets != 0).float().to(device)

            if tokenizer != None:
                targetsOrg = targets.clone()

                # encode with tokenizer and put to gpu
                inpts = tokenizerBatch(tokenizer, inpts, "encoding", device)
                targets = tokenizerBatch(tokenizer, targets, "encoding", device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                forward = model.forward(inpts, targets, training = True)
                forward = tokenizerBatch(tokenizer, forward, "decoding", device)
                forward = torch.reshape(forward, (forward.size(0), forward.size(1),  50, 50))
                loss = criterion(forward, targetsOrg)

            if tokenizer == None:
                # encode with tokenizer and put to gpu
                inpts = torch.flatten(inpts, start_dim = 2, end_dim = 3)
                targets = torch.flatten(targets, start_dim = 2, end_dim = 3)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                forward = model.forward(inpts, targets, training=True)

                loss = criterion(torch.flatten(forward, start_dim = 1, end_dim =2),
                                 torch.flatten(targets, start_dim = 1, end_dim = 2))


            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0) # gradient clipping; no exploding gradient
            optimizer.step()


            # save loss
            with torch.no_grad():
                if trainCounter % params["validationStep"] == 0 and trainCounter != 0:
                    #model.eval()
                    x, y = next(iter(valLoader))
                    x = x.float().to(device)
                    y = y.float().to(device)
                    y = (y != 0).float().to(device)

                    if tokenizer != None:
                        # encode with tokenizer and put to gpu
                        x = tokenizerBatch(tokenizer, x, "encoding", device)
                        y = tokenizerBatch(tokenizer, y, "encoding", device)

                        # predict
                        pred = model.forward(x,y, training = False)
                        pred = tokenizerBatch(tokenizer, pred, "decoding", device)
                        pred = torch.reshape(pred, (pred.size(0), pred.size(1),  50, 50))
                        valLoss = criterion(pred, yOrg)

                    if tokenizer == None:
                        # encode with tokenizer and put to gpu
                        x = torch.flatten(x, start_dim = 2, end_dim = 3)
                        y = torch.flatten(y, start_dim = 2, end_dim = 3)

                        # predict
                        pred = model.forward(x, y, training=False)

                        valLoss = criterion(torch.flatten(pred, start_dim=1, end_dim=2),
                                  torch.flatten(y, start_dim=1, end_dim=2))


                ## log to wandb
                if WandB:
                    wandb.log({"trainLoss": loss.detach().cpu().item(),
                            "validationLoss": valLoss.detach().cpu().item()})

                #save for csv
                trainLosses[trainCounter] = loss.detach().cpu().item()
                validationLosses[trainCounter] = valLoss.detach().cpu().item()
                epochs[trainCounter] = b
                trainCounter += 1

            # save model and optimizer checkpoint in case of memory overlow
            if trainCounter % 5000 == 0:
                saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)

                # save gradient descent
                df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
                df.to_csv(os.path.join(pathOrigin, modelName) + ".csv", index=False)

            # print loss
            print("epoch: ", b, ", example: ", trainCounter, " current loss = ", loss.detach().cpu().item())


    # save results of gradient descent
    path = os.path.join(pathOrigin, "/models")
    os.chdir(path)
    df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
    df.to_csv(modelName + ".csv", index=False)

    ## save model state
    saveCheckpoint(model, optimizer, modelName)
    print("results saved!")
    return


def trainLoopLSTM(trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin):
    """
    trains a given model on the data

    dataLoader: torch DataLoader object
    valLoader: torch DataLoader object
    tokenizer: nn.Module
        trained tokenizer, fixed in training
    model: torch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive with path
    params: dict
        lr, weightDecay, epochs, batchSize, validationStep, optimizer
    WandB: boolean
        use weights and biases tool to monitor losses dynmaically
    device: string
        device on which the data should be stored

    return: nn.class
        trained model and saved training monitoring data
    """
    # variables
    torch.autograd.set_detect_anomaly(True)
    trainLosses = np.ones(len(trainLoader) * params["batchSize"])
    validationLosses = np.ones(len(trainLoader) * params["batchSize"])
    epochs = np.ones(len(trainLoader) * params["batchSize"])
    trainCounter = 0
    valLoss = torch.zeros(1)
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config={
                "learning_rate": params["learningRate"],
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": params["epochs"],
            }
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])

    if params["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/models")
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/models/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    ###################### start training #############################

    model.train()
    for b in range(params["epochs"]):
        for inpts, targets in trainLoader:
            # use tokenizer on gpu
            inpts = inpts.to(device).float() # three maps, just use snow map as input
            targets = targets.to(device).float()
            model.train()

            # encode with tokenizer and put to gpu
            inpts = tokenizerBatch(tokenizer, inpts, "encoding", device)
            targets = tokenizerBatch(tokenizer, targets, "encoding", device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            forward = model.forward(inpts)
            loss = criterion(forward[0], targets) + forward[1]
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0) # gradient clipping; no exploding gradient
            optimizer.step()


            # save loss
            with torch.no_grad():
                if trainCounter % params["validationStep"] == 0 and trainCounter != 0:
                    x, y = next(iter(valLoader))
                    x = x.float()
                    y = y.float()
                    model.eval()

                    # encode with tokenizer and put to gpu
                    x = tokenizerBatch(tokenizer, x, "encoding", device)
                    y = tokenizerBatch(tokenizer, y, "encoding", device)

                    # predict
                    pred = model.forward(x)
                    valLoss = criterion(pred[0], y)


                ## log to wandb
                if WandB:
                    wandb.log({"trainLoss": loss.detach().cpu().item(),
                            "validationLoss": valLoss.detach().cpu().item()})

                #save for csv
                trainLosses[trainCounter] = loss.detach().cpu().item()
                validationLosses[trainCounter] = valLoss.detach().cpu().item()
                epochs[trainCounter] = b
                trainCounter += 1

            # save model and optimizer checkpoint in case of memory overlow
            if trainCounter % 5000 == 0:
                saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)

                # save gradient descent
                df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
                df.to_csv(os.path.join(pathOrigin, modelName) + ".csv", index=False)

            # print loss
            print("epoch: ", b, ", example: ", trainCounter, " current loss = ", loss.detach().cpu().item())


    # save results of gradient descent
    path = os.path.join(pathOrigin, "models")
    os.chdir(path)
    df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses, "epoch": epochs})
    df.to_csv(modelName + ".csv", index=False)

    ## save model state
    saveCheckpoint(model, optimizer, modelName)
    print("results saved!")
    return



def trainLoopConvLSTM(trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin):
    """
    trains a given model on the data

    dataLoader: torch DataLoader object
    valLoader: torch DataLoader object
    tokenizer: nn.Module
        trained tokenizer, fixed in training
    model: torch nn.class
    loadModel: boolean
    modelName: string
        .pth.tar model name on harddrive with path
    params: dict
        lr, weightDecay, epochs, batchSize, validationStep, optimizer
    WandB: boolean
        use weights and biases tool to monitor losses dynmaically
    device: string
        device on which the data should be stored

    return: nn.class
        trained model and saved training monitoring data
    """
    # variables
    torch.autograd.set_detect_anomaly(True)
    trainLosses = np.ones(len(trainLoader) * params["batchSize"])
    validationLosses = np.ones(len(trainLoader) * params["batchSize"])
    trainCounter = 0
    valLoss = torch.zeros(1)
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project= modelName,

            # track hyperparameters and run metadata
            config={
                "learning_rate": params["learningRate"],
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": params["epochs"],
            }
        )

    # get optimizer
    if params["optimizer"] == "adam":
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])
    if params["optimizer"] == "RMSProp":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                      lr = params["learningRate"],
                                      weight_decay= params["weightDecay"])

    # load model and optimizer from checkpoint
    if loadModel:
        # get into folder
        os.chdir(pathOrigin + "/models")
        lastState = loadCheckpoint(model, optimizer, pathOrigin + "/models/" + modelName)
        model = lastState[0]
        optimizer = lastState[1]

    ###################### start training #############################

    model.train()
    for b in range(params["epochs"]):
        for inpts, targets in trainLoader:
            # use tokenizer on gpu
            inpts = inpts.to(device).float() # three maps, just use snow map as input
            targets = targets.to(device).float()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize

            inpts = torch.permute(inpts, (1,0,2,3)).unsqueeze(dim = 2)
            forward = model.forward(inpts) ## teacher forcing??
            loss = criterion(forward, targets)
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0) # gradient clipping; no exploding gradient
            optimizer.step()
            trainCounter += 1

            # save loss
            with torch.no_grad():
                if trainCounter % params["validationStep"] == 0 and trainCounter != 0:
                    x, y = next(iter(valLoader))
                    x = x.to(device).float()
                    y = y.to(device).float()

                    # predict
                    x = torch.permute(x, (1,0,2,3)).unsqueeze(dim = 2)
                    pred = model.forward(x) ## teacher forcing??
                    valLoss = criterion(pred, y)


                ## log to wandb
                if WandB:
                    wandb.log({"trainLoss": loss.detach().cpu().item(),
                            "validationLoss": valLoss.detach().cpu().item()})

                #save for csv
                trainLosses[trainCounter] = loss.detach().cpu().item()
                validationLosses[trainCounter] = valLoss.detach().cpu().item()

            # save model and optimizer checkpoint in case of memory overlow
            if trainCounter % 5000 == 0:
                saveCheckpoint(model, optimizer, pathOrigin + "/" + "models/" + modelName)

                # save gradient descent
                df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses})
                df.to_csv(os.path.join(pathOrigin, modelName) + ".csv", index=False)

            # print loss
            print("epoch: ", b, ", example: ", trainCounter, " current loss = ", loss.detach().cpu().item())


    # save results of gradient descent
    path = os.path.join(pathOrigin, "/models")
    os.chdir(path)
    df = pd.DataFrame({"Train Loss": trainLosses, "Validation Loss": validationLosses})
    df.to_csv(modelName + ".csv", index=False)

    ## save model state
    saveCheckpoint(model, optimizer, modelName)
    print("results saved!")
    return

def getPatches(tensor, patchSize, stride=50):

    """
    takes an image and outputs list of patches in the image

    tensor: tensor
        input image
    patchSize: int
        x,y dimension of patch
    stride: int

    returns: list of tensor
        list of patches
    """
    # Get image dimensions
    nChannels, height, width = tensor.shape
    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1

    # Iterate over the patches and extract them
    patches = []
    counterX = 0
    counterY = 0
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            patch = tensor[:, counterX:counterX + patchSize, counterY:counterY + patchSize]
            # update counters
            counterX += stride

            # Add the patch to the list
            patches.append(patch)
        counterY += stride
        counterX = 0
    return patches


def combinePatches(patches, tensorShape, patchSize, stride=50, device= "cpu"):

    """
    combines a list of patches to full image

    patches: list of tensor
        patches in list
    tensorShape: tuple
        shape of output tensor
    patchSize: int
        x,y
    stride: int

    returns: tensor
        image in tensor reconstructed from the patches
    """
    # Get the number of channels and the target height and width
    n_channels, height, width = tensorShape

    # Initialize a tensor to store the image
    tensor = torch.zeros(tensorShape).to(device)

    # Calculate the number of patches in each direction
    nHorizontalPatches = (width - patchSize) // stride + 1
    nVerticalPatches = (height - patchSize) // stride + 1

    # Iterate over the patches and combine them
    patchIndex = 0
    counterX = 0
    counterY = 0
    for i in range(nVerticalPatches):
        for j in range(nHorizontalPatches):
            tensor[:, counterX:counterX + patchSize, counterY:counterY + patchSize] = patches[patchIndex]

            # update counters
            counterX += stride
            patchIndex += 1

        counterY += stride
        counterX = 0

    return tensor

"""
## test image functions
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/Helheim/filename.jpg"
img = Image.open(path)
convert_tensor = transforms.ToTensor()
t = convert_tensor(img)[:, 0:200, 0:200]
t_original = convert_tensor(img)[:, 0:200, 0:200]
print(t.size())
plt.imshow(np.transpose(t.numpy(), (1,2,0)))
plt.show()

t = getPatchesTransfer(t, 50, 40)
plt.imshow(np.transpose(t[0].numpy(), (1,2,0)))
plt.show()
print(len(t))
t = combinePatchesTransfer(t, (3, 200,200), 50, 40)
plt.imshow(np.transpose(t.numpy(), (1,2,0)))
plt.show()

print(t_original.numpy()- t.numpy())
"""

def createPatches(img, patchSize, stride):
    """
    creates image patches sampled from a region of interest

    img: np.array
    patchSize: int
        size of patches
    stride: int

    returns:  torch.tensor
        shape = (n_patches, bands, patchSize, patchSize)
    """
    # torch conversion, put into ndarray
    img = torch.from_numpy(img)
    patches = getPatches(img, patchSize, stride=stride)
    out = torch.stack(patches, dim = 0)
    out = out.numpy()

    return out


def automatePatching(data, patchSize, stride):
    """
    creates image patches sampled from a region of interest

    data: list of tuple of datetime and np.array
        data extracted from API
    patchSize: int
        size of patches
    maxPatches: int
        number of patches extracted

    returns:  list of tuple of datetime and np.array
        switch np array in tuple with np array with on more dimension -> patches
    """

    res = []
    for i in range(len(data)):
        print("patchify scene: ", i)
        patches = createPatches(np.expand_dims(data[i], axis=0), patchSize, stride)
        res.append(patches)

    return res

def imageAlignArcosics(img, reference):
    """
    aligns the array to the reference array using fourier space transformations

    img: np.array
    reference: np.array
    return: np.array
        aligned array to the input
    """

    # convert to geoARray
    img = GeoArray(img, projection= "EPSG:32643")
    reference = GeoArray(reference, projection= "EPSG:32643" )

    # parameters
    kwargs = {
        'grid_res': 50,
        'window_size': (50, 50),
        'path_out': '/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets',
        'projectDir': 'my_project',
        'q': False,
    }

    # apply
    CR = COREG_LOCAL(img, reference, **kwargs)
    #CR = COREG(img, reference, align_grids=True, ws = (50, 50))
    res = CR.correct_shifts()

    return res["arr_shifted"]


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

def monthlyAverageScenesEnCC(d, ROI, applyKernel):

    """
    gets list of list with scenes from  different months and averages over every two months with the aligned scenes

    d: list of tuple of datetime and np.array
    ROI: list of int
        region of interest to be processed
    applyKernel: boolean

    returns: list of list of torch.tensor
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



def monthlyAverageScenes(d, ROI, applyKernel):

    """
    gets list of list with scenes from  different months in a year and returns monthly averages with missing images interpolated in between
    d: list of tuple of datetime and np.array

    ROI: list of int
        region of interest to be processed
    applyKernel: boolean

    returns: list of list of torch.tensor
    """
    # get region of interest in data

    for i in range(len(d)):
        d[i] = (d[i][0] ,d[i][1][:, ROI[0]:ROI[1], ROI[2]:ROI[3]])

    # check for the correct months to make data stationary -> summer data
    l = []
    counterImg = 0
    usedMonths = []
    months =  [[i, i+1] for i in range(1, 13, 2) if i+1 <= 12]

    for y in np.arange(2013, 2022, 1): # year
        #for m in np.arange(1,13,1): # month
        for [m, t] in months:
            imgAcc = np.zeros((ROI[1] - ROI[0], ROI[3] - ROI[2]))
            month = 0
            for i in range(len(d)):
                if (((convertDatetoVector(d[i][0])[1].item() == m) or (convertDatetoVector(d[i][0])[1].item() == t))  and (convertDatetoVector(d[i][0])[2].item() == y)):
                    # count months
                    month += 1

                    ## get roi and apply kernel
                    img = d[i][1][[2,5], :, :]

                    # clean missings with kernel
                    if applyKernel:
                        # apply kernel to raw bands, do many times as sometimes division by zero gives nans in image
                        for z in [0, 1]:  # only use NDSI relevant bands
                            while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                                img[z, :, :] = applyToImage(img[z, :, :])
                                print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
                            #img[z, :, :] = imageAlignArcosics(img[z, :, :], ref[z, :, :])
                            print("band: ", z, " of ", img.shape[0], "done")
                    print("application of kernel done")

                    # apply NDSI here
                    # add snow mask to average image
                    threshold = 0.3
                    NDSI = np.divide(np.subtract(img[0, :, :], img[1, :, :]),
                                 np.add(img[0, :, :], img[1, :, :]))
                    #nosnow = np.ma.masked_where(NDSI >= threshold, NDSI).filled(0) ## leave in case necessary to use in future
                    snow = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)

                    # align
                    if month == 1:
                        imgAcc += snow
                    if month > 1:
                        # align before averaging
                        snow = alignImages(snow, imgAcc, 5000, 0.1)
                        #snow = imageAlignArcosics(snow, imgAcc)

                        plt.imshow(snow)
                        plt.show()
                        imgAcc = (imgAcc + snow) / 2 # average


            if month != 0:
                # average over images
                #imgAcc = imgAcc / month
                l.append(imgAcc)

            ## mark months with no scenes
            if month  == 0:
                l.append(np.full((ROI[1]-ROI[0], ROI[3]-ROI[2]), np.nan))
                counterImg += 1
                print(counterImg, " missing imgs")

            usedMonths.append(np.array([[m,y]]))

        print("averaging of year: ", y, " done")

    # interpolate missing scenes for months
    print("start image interpolation")

    # sanity check
    assert len(usedMonths) == len(l) == 9*12 # 9 years, 12 months


    # interpolate between images
    ## first images can not be interpolated take first not missing image as template and impute
    indices = [i for i in range(len(l)) if not np.any(np.isnan(l[i]))]

    for i in range(len(indices)-1):
        idx = indices[i]
        succ = indices[i+1]

        if succ - idx == 1:
            pass

        elif succ - idx > 1:
            diff = succ - idx
            delta = (l[succ] - l[idx])/diff
            for t in range(diff):
                #l[idx + t + 1] = l[idx] + (t+1) * delta
                l[idx + t + 1] = np.ma.masked_where(l[idx] + (t+1) * delta < threshold, NDSI).filled(0)

    arr = np.transpose(np.dstack(l), (2, 0, 1))
    print("interpolation done")
    result = [arr[i, :, :] for i in range(arr.shape[0])]

    ## save on harddrive
    print("start saving scenes")
    path = os.getcwd()
    os.makedirs("monthlyAveragedScenes", exist_ok=True)
    os.chdir(os.path.join(path, "monthlyAveragedScenes"))
    os.makedirs("images", exist_ok=True)
    os.makedirs("dates", exist_ok=True)
    counter = 0

    # get beginning and end missings
    indices = [i for i in range(len(result)) if not np.any(np.isnan(result[i]))]

    result = [result[i] for i in indices]
    usedMonths = [usedMonths[i] for i in indices]

    # align images
    result = aligneOverTime(result)

    # crop edges in case of shifting
    result = [result[i][50:750, 50:750] for i in range(len(result))]

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

def getTrainTest(patches, window, inputBands, outputBands, stationary):
    """
    converts patches to image data for deep learning models

    patches: list of tensor
        data createPatches.py
    window: int
        length of sequences for model
    inputBands: list of int
    outputBands: list of int
    stationary: boolean
        quantized time

    returns: list of list of input data, input date and target data, target date

    """
    #Path = os.getcwd()
    Path = "/home/jonas/datasets" ## change here to where train data is saved
    if stationary: # yearly data; too sparse
        years = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]

        # get list of consecutive patches in same time of year per year in list -> list of lists
        listPatches = []
        for year in years:
            helper = []
            for b in range(len(patches)):
                if convertDatetoVector(patches[b][0])[2].item() == float(year):
                    helper.append(patches[b])
            print(len(helper))
            listPatches.append(helper)
        counter = 0

        for i in range((len(listPatches) - 2 * window) // 1 + 1):  # formula from pytorch cnn classes, move over years
            for o in range(20): # hard coded, sample 50 consecutive scene lists with 2* window elements
                x = []
                y = []
                ## take next n scenes raondomly from consecutive years
                for d in range(window):
                    l = len(listPatches[d])
                    e = len(listPatches[d+window])
                    sceneX = listPatches[d][np.random.randint(l)]
                    sceneY = listPatches[d+window][np.random.randint(e)]
                    x.append(sceneX)
                    y.append(sceneY)

                for z in range(x[0][1].shape[0]): # iterate over patches in list
                    xDates = [convertDatetoVector(x[t][0]) for t in range(len(x))]
                    xDates = torch.stack(xDates, dim=0)
                    yDates = [convertDatetoVector(y[t][0]) for t in range(len(y))]
                    yDates = torch.stack(yDates, dim=0)
                    xHelper = list(map(lambda x: torch.from_numpy(x[1][z, inputBands, :, :]), x))
                    xHelper = torch.stack(xHelper, dim=0)
                    yHelper = list(map(lambda x: torch.from_numpy(x[1][z, outputBands, :, :]), y))
                    yHelper = torch.stack(yHelper, dim=0)

                    counter += 1
                    # sanity checks
                    assert len(xDates) == len(yDates) == len(xHelper) == len(yHelper) == window
                    assert xDates[0][2] < xDates[1][2] < xDates[2][2] < xDates[3][2] # delta t correct?
                    assert yDates[0][2] < yDates[1][2] < yDates[2][2] < yDates[3][2]

                    # just save images and targets in folder
                    # input
                    os.chdir(Path)
                    os.makedirs("images", exist_ok=True)
                    os.chdir(os.path.join(os.getcwd(), "images"))

                    # save data object on drive
                    with open(str(counter), "wb") as fp:  # Pickling
                        pickle.dump(xHelper, fp)

                    # targets
                    os.chdir(Path)
                    os.makedirs("targets", exist_ok=True)
                    os.chdir(os.path.join(os.getcwd(), "targets"))

                    # save data object on drive
                    with open(str(counter), "wb") as fp:  # Pickling
                        pickle.dump(yHelper, fp)

    elif stationary == False:
        counter =  0 # 37633 for mixed dataset
        for i in range((len(patches) - 2*window) // 1 + 1): # formula from pytorch cnn classes
            # create patches from random consecutive timepoints in the future
            ## take next n scenes
            x = patches[i:i + window]
            y = patches[i + window: i + (2 * window)]
            for z in range(x[0].shape[0]):

                xHelper = list(map(lambda x: torch.from_numpy(x[z, inputBands, :, :]), x))
                xHelper = torch.stack(xHelper, dim = 0)
                yHelper = list(map(lambda x: torch.from_numpy(x[z, outputBands, :, :]), y))
                yHelper = torch.stack(yHelper, dim=0)

                # sanity checks
                assert len(xHelper) == len(yHelper)

                # just save images and targets in folder
                # input
                os.chdir(Path)
                os.makedirs("images", exist_ok=True)
                os.chdir(os.path.join(os.getcwd(), "images"))

                # save data object on drive
                with open(str(counter), "wb") as fp:  # Pickling
                    pickle.dump(xHelper, fp)

                # targets
                os.chdir(Path)
                os.makedirs("targets", exist_ok=True)
                os.chdir(os.path.join(os.getcwd(), "targets"))

                # save data object on drive
                with open(str(counter), "wb") as fp:  # Pickling
                    pickle.dump(yHelper, fp)
                counter += 1

    return

# input 5, 3, 50, 50; targets: 5, 1, 50, 50
def fullSceneLoss(inputScenes, inputDates, targetScenes, targetDates,
                  model, patchSize, stride, outputDimensions, device = "cpu", training = True,
                  test = False):
    """
    train model on loss of full scenes and backpropagate full scene error in order to get smooth boarders in the final scene predictions

    inputScenes: tensor
        scenes
    inputDates: tensor
        dates
    targetScenes: tensor
        target scenes
    targetDates: tensor
        target dates
    model: torch.model object
    patchSize: int
    stride: int
        used stride for patching
    outputDimensions: tuple
        dimensions of output scenes
    device: string
        on which device is tensor calculated
    training: boolean
        inference?
    test: boolean
        test pipeline without model predictions


    returns: int
        loss on full five scenes and all associated patches

    """

    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(inputScenes.size(0)):
        helper = getPatches(inputScenes[i], patchSize, stride)
        inputList.append(helper)

        helper = getPatches(targetScenes[i], patchSize, stride)
        targetList.append(helper)

    # get predictions from input patches
    if test == False:
        latentSpaceLoss = 0
        for i in range(len(inputList[0])):
            helperInpt = list(x[i] for x in inputList)
            targetInpt = list(x[i] for x in targetList)
            inputPatches = torch.stack(helperInpt, dim = 0)
            targetPatches = torch.stack(targetInpt, dim=0)


            # put together for final input
            finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

            # predict with model
            prediction = model.forward(finalInpt, training = training)

            # switch input with predictions; z = scene index, i = patch index
            for z in range(prediction[0].size(0)):
                inputList[z][i] = prediction[0][z, :, :]

            # accumulate latent space losses
            latentSpaceLoss += prediction[1].item()

        # get final loss of predictions of the full scenes
        # set patches back to images
        scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride, device = device) for x in inputList)
        fullLoss = sum(list(map(lambda x,y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        fullLoss += latentSpaceLoss


        # save memory
        #del prediction
        #del scenePredictions

        return fullLoss

    if test:
        scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)
        fullLoss = sum(list(map(lambda x, y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        return fullLoss

        latentSpaceLoss = 0
        for i in range(len(inputList[0])):
            helperInpt = list(x[i] for x in inputList)
            targetInpt = list(x[i] for x in targetList)
            inputPatches = torch.stack(helperInpt, dim=0)
            targetPatches = torch.stack(targetInpt, dim=0)

            # use targets in order to test pipeline without model prediction, takes 5 images extracts pathces and puts images back again
            # without model predictions loss should be 0 if pipeline works
            for z in range(inputPatches.size(0)):
                inputList[z][i] = targetPatches[z, :, :]


        # get final loss of predictions of the full scenes
        # set patches back to images
        scenePredictions = list(combinePatchesTransfer(x, outputDimensions, patchSize, stride) for x in inputList)
        fullLoss = sum(list(map(lambda x, y: nn.MSELoss()(x, y), scenePredictions, targetScenes)))
        return fullLoss


def fullSceneTrain(model, modelName, optimizer, data, epochs, patchSize, stride, outputDimensions, device,
                   WandB, pathOrigin = pathOrigin):
    """

    train model on full scenes

    model: torch nn.model
    modelName: string
    optimizer: torch optim object
    data: list of list of tensor, tensor and tensor, tensor
        five scenes input and dates, five scenes targets and dates
    epochs: int
    patchSize: int
    stride: int
    outputDimensions: tuple
    device: string
        machine to be used
    pathOrigin: str
        path for data safing

    """
    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project=modelName,

            # track hyperparameters and run metadata
            config={
                "architecture": modelName,
                "dataset": "Helheim, Aletsch, jakobshavn",
                "epochs": epochs,
            }
        )


    trainCounter = 0
    runningLoss = 0
    trainLosses = []
    for x in range(epochs):
        # get indices for epoch
        ix = np.arange(0, len(data), 1)
        ix = np.random.choice(ix, len(data), replace=False, p=None)

        for i in ix:
            # get data
            helper = data[i]

            # move to cuda
            helper = moveToCuda(helper, device)

            y = helper[1][0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # use only target prediction for loss, latent space should already be learned, just edges of patches are smoothed
            loss = fullSceneLoss(helper[0][0], helper[0][1],
                                 helper[1][0], helper[1][1],
                                 model,
                                 patchSize,
                                 stride,
                                 outputDimensions,
                                 device=device,
                                 training = True,
                                 test = False)
            loss = torch.divide(loss, 144) # normalize loss
            loss.backward()
            optimizer.step()
            trainCounter += 1

            # print loss
            runningLoss += loss.item()
            meanRunningLoss = runningLoss / trainCounter
            trainLosses.append(meanRunningLoss)

            ## log to wandb
            if WandB:
                wandb.log({"train loss": meanRunningLoss})

            # save memory
            #del loss

            print("epoch: ", x, ", example: ", trainCounter, " current loss = ", meanRunningLoss)

    path = pathOrigin + "/results"
    os.chdir(path)
    os.makedirs(modelName, exist_ok=True)
    os.chdir(path + "/" + modelName)

    ## save model
    saveCheckpoint(model, modelName)

    # save losses
    dict = {"trainLoss": trainLosses}
    trainResults = pd.DataFrame(dict)

    # save dataFrame to csv
    trainResults.to_csv("resultsTrainingScenes.csv")

    return

## visualize network performance on full scenes, use for testData, qualitative check
def inferenceScenes(model, data, patchSize, stride, outputDimensions, glacierName, predictionName, modelName, device, plot = False, safe = False, pathOrigin = pathOrigin):
    """
    use for visual check of model performance

    model: nn.class object
    data: same as above
    patchSize: int
    stride: int
    outputDimensions: tuple
    glacierName: str
        name of the glacier for order structure
    predictionname: str
        name of folder for predictions to be safed in
    modelName: string
        name of the model to safe in order structure
    device: sring
        machine to compute on
    plot: boolean
    safe: boolean
        safe output as images on harddrive

    return: list of tensor
        predicted scenes
    """
    # inference mode
    model.eval()
    # move to cuda
    data = moveToCuda(data, device)

    inputScenes = data[0][0]
    targetScenes = data[1][0]
    inputDates = data[0][1]
    targetDates = data[1][1]


    # get patches from input images and targets
    inputList = []
    targetList = []
    for i in range(inputScenes.size(0)):
        helper = getPatches(inputScenes[i], patchSize, stride)
        inputList.append(helper)

        helper = getPatches(targetScenes[i], patchSize, stride)
        targetList.append(helper)

    # get predictions from input patches
    for i in range(len(inputList[0])):
        helperInpt = list(x[i] for x in inputList)
        targetInpt = list(x[i] for x in targetList)
        inputPatches = torch.stack(helperInpt, dim=0)
        targetPatches = torch.stack(targetInpt, dim=0)

        # put together for final input
        finalInpt = [[inputPatches, inputDates], [targetPatches, targetDates]]

        # predict with model
        prediction = model.forward(finalInpt, training=False)

        # switch input with predictions; z = scene index, i = patch index
        for z in range(prediction.size(0)):
            inputList[z][i] = prediction[z, :, :]

    # get final loss of predictions of the full scenes
    # set patches back to images
    scenePredictions = list(combinePatches(x, outputDimensions, patchSize, stride) for x in inputList)

    ## plot
    if plot:
        plotList = [data[0][0][d] for d in range(5)]
        plotList = plotList + scenePredictions
        plotList = [x.detach().cpu().numpy() for x in plotList]
        plotList = [np.transpose(x, (1,2,0)) for x in plotList]

        fig, axs = plt.subplots(2, 5)

        for i in range(10):
            ax = axs[i // 5, i % 5]
            ax.imshow(plotList[i])
            ax.axis('off')

        plt.show()
    if safe:
        print("start saving prediction scenes")

        path = pathOrigin + "/results"
        os.chdir(path)
        os.makedirs(modelName, exist_ok=True)
        os.chdir(path + "/" + modelName)

        os.makedirs("modelPredictions", exist_ok=True)
        os.chdir(os.getcwd() + "/modelPredictions")
        os.makedirs(glacierName, exist_ok=True)
        os.chdir(os.getcwd() + "/" + glacierName)
        os.makedirs(predictionName, exist_ok=True)
        os.chdir(os.getcwd() + "/" + predictionName)

        path = os.getcwd()
        for i in range(len(scenePredictions)):
            # model predictions
            os.chdir(path)
            os.makedirs("predictions", exist_ok=True)
            os.chdir(path + "/" + "predictions")
            plt.imshow(minmaxScaler(scenePredictions[i].cpu().detach().numpy()[0,:,:]), cmap='gray')

            # save on harddrive
            p = os.getcwd()+ "/" + str(i) + ".pdf"
            plt.savefig(p, dpi=1000)

            with open(str(i), "wb") as fp:  # Pickling
                pickle.dump(scenePredictions[i].cpu().detach().numpy(), fp)


            # target predictions
            os.chdir(path)
            os.makedirs("targets", exist_ok=True)
            os.chdir(path + "/" + "targets")
            plt.imshow(minmaxScaler(targetScenes[i].cpu().detach().numpy()[0, :, :]), cmap='gray')

            # save on harddrive
            p = os.getcwd() + "/" + str(i) + ".pdf"
            plt.savefig(p, dpi=1000)

            with open(str(i), "wb") as fp:  # Pickling
                pickle.dump(targetScenes[i].cpu().detach().numpy(), fp)

    print("prediction scenes saved")
    #return scenePredictions
    return


def moveToCuda(y, device):
    """
    transfers datum to gpu/cpu

    y: list of list of tensor and tensor and list of tensor and tensor
        input datum
    return: list of list of tensor and tensor and list of tensor and tensor
        transferred to cuda gpu
    """

    y[0][0] = y[0][0].to(device).to(torch.float32).requires_grad_()
    y[0][1] = y[0][1].to(device).to(torch.float32).requires_grad_()
    y[1][0] = y[1][0].to(device).to(torch.float32).requires_grad_()
    y[1][1] = y[1][1].to(device).to(torch.float32).requires_grad_()

    return y


def loadFullSceneData(path, names, window, inputBands, outputBands, ROI, applyKernel):
    """
    creates dataset of full scenes in order to train model on full scene loss

    path: string
        path to dataset
    names: list of string
        names of files to load
    inputBands: int
        number of the input to be used
    outputBands: int
        number of bands to be used in the output

    return: list of tensor and tensor, and tensor and tensor
        datum = list of scenes, dates and targets and dates
    """
    d = loadData(path, names)


    # crop to ROIs
    for i in range(len(d)):
        d[i] = (d[i][0], d[i][1][:, ROI[0]:ROI[1], ROI[2]:ROI[3]])

    # kernel for nans
    if applyKernel:
        for i in range(len(d)):
            img = d[i][1]
            for z in [2,5]:
                while np.count_nonzero(np.isnan(img[z, :, :])) > 0:
                    img[z, :, :] = applyToImage(img[z, :, :])
                    print("still missing ", np.count_nonzero(np.isnan(img[z, :, :])), " pixels")
                print("band: ", z, " of ", img.shape[0], "done")
            d[i] = (d[i][0], NDSI(img, 0.3)) # add NDSI here
    print("application of kernel done")

    dataList = []
    deltas = np.arange(1, 6, 1)  # [1:5]
    counter = 0
    for delta in deltas:
        sceneList = d[::delta]
        for i in range((len(sceneList) - 2 * window) // 1 + 1):  # formula from pytorch cnn classes
            # create patches from random consecutive timepoints in the future
            ## take next n scenes
            x = sceneList[i:i + window]
            y = sceneList[i + window: i + (2 * window)]

            # dates
            xDates = [convertDatetoVector(x[t][0]) for t in range(len(x))]
            xDates = torch.stack(xDates, dim=0)
            yDates = [convertDatetoVector(y[t][0]) for t in range(len(y))]
            yDates = torch.stack(yDates, dim=0)

            # ROI in scenes
            xHelper = list(map(lambda x: torch.from_numpy(x[1][inputBands, :, :]), x))
            xHelper = torch.stack(xHelper, dim=0)
            yHelper = list(map(lambda x: torch.from_numpy(x[1][outputBands, :, :]), y))
            yHelper = torch.stack(yHelper, dim=0).unsqueeze(1)

            # sanity checks
            assert len(xDates) == len(yDates) == len(xHelper) == len(yHelper)

            # save
            dataList.append([[xHelper, xDates], [yHelper, yDates]])

        print("delta ", counter, " done")
        counter += 1

    # save data object on drive
    with open("trainDataFullScenes", "wb") as fp:  # Pickling
        pickle.dump(dataList, fp)
    print("data saved!")

    return dataList

## test
"""
path = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn"
dat = loadFullSceneData(path, ["2013", "2014"], 5, [7,8,9], 9, [50, 650, 100, 700], True)

# ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
os.chdir("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets")
# save data object on drive
with open("test", "wb") as fp:  # Pickling
    pickle.dump(dat, fp)
print("data saved!")
"""


def plotPatches(model, data, transformer, tokenizer, device, plot):
    """
    plots patches and targets and saves on harddrive

    model: nn.model object
    data: list of tensor
    transformer: boolean
    tokenizer: nn.object module
    plot: boolean
    """


    model.eval()

    # predictions
    if transformer:
        x = tokenizerBatch(tokenizer, data[0].to(device).float(), "encoding", device)
        #y = tokenizerBatch(tokenizer, data[1], "encoding", device)

        # forward + backward + optimize
        forward = model.forward(x,None, training=False)
        forward = tokenizerBatch(tokenizer, forward, "decoding", device)
        forward = torch.reshape(forward, (forward.size(0), forward.size(1), 50, 50))


    predictions = forward
    targets = data[1].to(device).float()

    # put into list
    predList = []
    targetList = []
    for i in range(4):

        pred = predictions[:, i, :, :].detach().cpu().numpy().squeeze()
        predList.append(pred)

        targ = targets[:, i, :, :].detach().cpu().numpy().squeeze()
        targetList.append(targ)

    plotData = predList + targetList


    # integrate dates here

    # check inference
    assert len(plotData) == 8

    # start plotting
    path = pathOrigin + "/predictions"
    os.chdir(path)
    name = str(np.random.randint(50000))
    os.makedirs(name, exist_ok=True)
    os.chdir(path + "/" + name)

    path = os.getcwd()
    for i in range(len(plotData)):
        # model predictions
        plt.imshow(minmaxScaler(plotData[i]), cmap='gray')
        plt.axis('off')
        # save on harddrive
        p = os.getcwd() + "/" + str(i) + ".pdf"
        plt.savefig(p, dpi=1000)

        with open(str(i), "wb") as fp:  # Pickling
            pickle.dump(plotData[i], fp)

    # Show the plot
    if plot:
        plt.show()

    return

def getTrainDataTokenizer(paths):
    """
    saves each patch created as 50x50 tensor

    paths: list of str
        paths to patches and targets
    """
    counter = 0
    for path in paths:
        ## images folder
        # get number of patches
        images = os.listdir(os.path.join(path, "images"))
        pathsImg = [os.path.join(os.path.join(path, "images"), item) for item in images]

        for imgPath in pathsImg:
            tensor = openData(imgPath)
            for i in range(tensor.size(0)):
                #img = tensor[i, 2, :, :]
                img = tensor[i,:,:]

                # save into folder
                currentPath = os.getcwd()
                outputPath = "/home/jonas/datasets/parbati/tokenizer"
                os.chdir(outputPath)
                with open(str(counter), "wb") as fp:  # Pickling
                    pickle.dump(img, fp)
                os.chdir(currentPath)
                counter += 1
                if counter % 1000 == 0:
                    print("image: ", counter, " done")
        print("path: ", path, " images done")

        ## targets folder
        # get number of patches
        targets = os.listdir(os.path.join(path, "targets"))
        pathsImg = [os.path.join(os.path.join(path, "targets"), item) for item in targets]

        for imgPath in pathsImg:
            tensor = openData(imgPath)
            for i in range(tensor.size(0)):
                img = tensor[i, :, :]

                # save into folder
                currentPath = os.getcwd()
                outputPath = "/home/jonas/datasets/parbati/tokenizer"
                os.chdir(outputPath)
                with open(str(counter), "wb") as fp:  # Pickling
                    pickle.dump(img, fp)
                os.chdir(currentPath)
                counter += 1
                if counter % 1000 == 0:
                    print("image: ", counter, " done")
        print("path: ", path, " targets done")
        #if path == "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/patched":
        #    quit()

    return
"""
d = [ #"/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched"] #,
     #"/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jakobshavn/patched"]#,
     "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Jungfrau_Aletsch_Bietschhorn/patched"]
"""
"""
d = ["/home/jonas/datasets/parbati"]
getTrainDataTokenizer(d)
"""




















