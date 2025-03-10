import os
from torch.utils.data import Dataset
from torch import is_tensor
import functions
from torch.utils.data import DataLoader
import numpy as np

class glaciers(Dataset):
    def __init__(self, path, mode, bootstrap = False):
        """
        dataset class for train loop
        path: str
            path to image and target folder
        mode: str
            train, val, test
        """
        self.mode = mode
        self.path = path
        self.bootstrap = bootstrap

        # take 80% of data as train and val, take 20% of data for testing
        # get list of all image paths in directory
        images = os.listdir(os.path.join(self.path, "images"))
        paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
        # take 80% of data as training
        criterion = round(len(paths) * 0.80)
        paths = paths[0:criterion]
        self.images = paths

        targets = os.listdir(os.path.join(self.path, "targets"))
        paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
        paths = paths[0:criterion]
        self.targets = paths


        if self.mode == "train":
            # take 90% of data as training
            criterion = round(len(self.images) * 0.90)
            self.images = self.images[0:criterion]
            self.targets = self.targets[0:criterion]

        if self.mode == "val":
            # take 10% as validation set
            criterion = round(len(paths) * 0.9)
            self.images = self.images[criterion:]
            self.targets = self.targets[criterion:]

        if self.mode == "test":
            # get list of all image paths in directory
            images = os.listdir(os.path.join(self.path, "images"))
            paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
            criterion = round(len(paths) * 0.8)
            paths = paths[criterion:]
            self.images = paths

            targets = os.listdir(os.path.join(self.path, "targets"))
            paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
            paths = paths[criterion:]
            self.targets = paths


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()

        if self.bootstrap == False:
            try:
                # get data in tensor format
                inpt = functions.openData(self.images[idx])
                target = functions.openData(self.targets[idx])
            except:
                # get data in tensor format
                index = np.random.randint(self.__len__())
                inpt = functions.openData(self.images[index])
                target = functions.openData(self.targets[index])

        if self.bootstrap == True:
            try:
                # get data in tensor format
                index = np.random.randint(self.__len__())
                inpt = functions.openData(self.images[index])
                target = functions.openData(self.targets[index])
            except:
                # get data in tensor format
                index = np.random.randint(self.__len__())
                inpt = functions.openData(self.images[index])
                target = functions.openData(self.targets[index])

        return inpt, target


class tokenizerData(Dataset):
    def __init__(self, path):
        """
        dataset class for train loop
        path: str
            path to image folder
        """
        self.path = path

        # get list of all image paths in directory
        images = os.listdir(os.path.join(self.path, "images"))
        paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
        criterion = round(len(paths) * 0.80) # only take trainingData
        paths = paths[0:criterion]
        self.images = paths

        targets = os.listdir(os.path.join(self.path, "targets"))
        paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
        paths = paths[0:criterion]
        self.targets = paths

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()
        try:
            # flip coin if targets or images
            flip = np.random.randint(0, 2, size=1)[0]

            # get data in tensor format
            if flip == 0:
                inpt = functions.openData(self.images[idx])[np.random.randint(0, 4, size=1)[0], :, :]
            if flip == 1:
                inpt = functions.openData(self.targets[idx])[np.random.randint(0, 4, size=1)[0], :, :]

        except:
            # get data in tensor format
            index = np.random.randint(self.__len__())
            inpt = functions.openData(self.targets[index])[np.random.randint(0, 4, size=1)[0], :, :]

        return inpt
"""
training_data = glaciers("/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code/datasets/Helheim/patched")
train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)

train_features, train_labels = next(iter(train_dataloader))
print(train_features)
"""
