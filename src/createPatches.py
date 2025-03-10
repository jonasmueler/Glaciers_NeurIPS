from alignment import *
from dataAPI import *
from config import *
import torch


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

    data: list of np.array
    patchSize: int
        size of patches
    stride: int
        moving window regulation factor

    returns: list of list of np.array
        list of list of patches from each image 
        
    """

    res = []
    for i in range(len(data)):
        print("patching scene: ", i)
        patches = createPatches(np.expand_dims(data[i], axis=0), patchSize, stride)
        res.append(patches)

    return res


def getTrainTest(patches, window, inputBands, outputBands):
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
    # create folder to save output 
    os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched"), exist_ok = True)
    os.chdir(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched"))

    # start generating data
    counter =  0 
    for i in range((len(patches) - 2*window) // 1 + 1): # formula from pytorch cnn classes
        # create patches from consecutive timepoints in the future
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
            
            os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched", "images"), exist_ok=True)
            os.chdir(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched", "images"))

            # save data object on drive
            with open(str(counter), "wb") as fp:  # Pickling
                pickle.dump(xHelper, fp)

            # targets
            os.makedirs(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched", "targets"), exist_ok=True)
            os.chdir(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched", "targets"))

            # save data object on drive
            with open(str(counter), "wb") as fp:  # Pickling
                pickle.dump(yHelper, fp)
            counter += 1

    return None


def main():
    # load data
    currentPath = os.path.join(path, "datasets", name, "alignedAveragedDataNDSI")
    os.chdir(currentPath)
    files = glob.glob(os.path.join(currentPath, '*'))
    data = []
    for i in range(len(files)):
        data.append(openData(files[i]))
    print("data loaded")

    # create patches
    d = automatePatching(data, patchSize, stride)
    print("data patched")

    # put into tensors
    d = getTrainTest(d, sequenceLength, 0,0)
    print("data converted to tensors and saved")

    return None 

if __name__ == "__main__":
    main()