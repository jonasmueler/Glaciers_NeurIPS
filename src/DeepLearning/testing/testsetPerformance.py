import pandas as pd
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import numpy as np
import matplotlib.pyplot as plt
from unet_model import UNet
import transformerBase
import LSTM
import ConvLSTM
import lstmAttention
from unet_model import UNet

## global variables for project
### change here to run on cluster ####
pathOrigin = "/mnt/qb/work/ludwig/lqb875"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"
tokenizer = False

if tokenizer:
    # load tokenizer and model
    Tokenizer = tokenizer.tokenizer()
    os.chdir(pathOrigin + "/models")
    Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
    Tokenizer = Tokenizer.to(device)


#model = ConvLSTM.ConvLSTMPredictor([64, 64, 24, 24, 64, 24]).to(device)
model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = UNet(1,1).to(device)


# load weights to transformers
model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMAttentionSmall"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "Unet"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "ConvLSTM"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMEncDec"))
print("loading models finished")

# dataLoader /home/jonas/datasets/parbati
datasetTest = datasetClasses.glaciers(os.path.join(pathOrigin, "datasets", "parbati"), "test", bootstrap = False)
#datasetTest = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test", bootstrap = True)
dataTest = DataLoader(datasetTest, 100, shuffle = True)

with torch.no_grad():
    # do 2000 bootstrap iterations
    losses = []
    nIterations = 1
    MSELoss = torch.nn.MSELoss()
    MAELoss = torch.nn.L1Loss()
    modelName = "LSTMAttentionSmall"
    modelResults = np.zeros((nIterations, 2))

    for b in range(nIterations):
        MSElosses = torch.zeros(len(dataTest))
        MAElosses = torch.zeros(len(dataTest))
        counter = 0

        # check model performance on bootstrapped testset
        for inpts, targets in dataTest:

            inpts = inpts.to(device).float()
            targets = targets.to(device).float().squeeze()
            #inpts = torch.flatten(inpts, start_dim=2, end_dim=3)
            #targets = torch.flatten(targets, start_dim=2, end_dim=3)

            if tokenizer:
                # encode with tokenizer and put to gpu
                inpts = functions.tokenizerBatch(Tokenizer, inpts, "encoding", device)
                targets = functions.tokenizerBatch(Tokenizer, targets, "encoding", device)

            # predict
            model.eval()
            forward = model.forward(inpts, targets, training=False)

            if tokenizer:
                forward = Tokenizer.decoder(forward)
                forward = functions.tokenizerBatch(Tokenizer, forward, "decoding", device)
                forward = torch.reshape(forward, (1, forward.size(0), 50, 50))

            # get loss
            MSE = MSELoss(forward, targets)
            MAE = MAELoss(forward, targets)
            MSElosses[counter] = MSE
            MAElosses[counter] = MAE
            counter += 1

        MSE = torch.mean(MSElosses)
        MAE = torch.mean(MAElosses)
        modelResults[b, 0] = MSE.detach().cpu().numpy()
        modelResults[b, 1] = MAE.detach().cpu().numpy()

    # save model results
    df = pd.DataFrame(modelResults, columns = ["MSE", "MAE"])
    os.chdir(os.path.join(pathOrigin, "models"))
    df.to_csv(modelName + "_bootstraped_results" + ".csv")
    os.chdir(pathOrigin)







