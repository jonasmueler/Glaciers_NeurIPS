from alignment import *

def enhancedCorAlign(imgStack, bands = [2,5,5]):
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
        img = d[i][1][[2,5], ROI[0]:ROI[1], ROI[2]:ROI[3]]
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

    # NDSI
    print("calculate NDSI masks")        
    threshold = 0.3
    for i in range(alignedData.shape[0]):
        NDSI = np.divide(np.subtract(alignedData[i, 0, :, :], alignedData[i, 1, :, :]),
                                         np.add(alignedData[i, 0, :, :], alignedData[i, 1, :, :]))
        alignedData[i] = np.ma.masked_where(NDSI < threshold, NDSI).filled(0)

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

#if __name__  == "__main__":

