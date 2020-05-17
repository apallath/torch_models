import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
import cv2

class Cell_dataset(Dataset):
    """Pocked Cell dataset."""
    def __init__(self, file_location):
        self.location = file_location
        self.filenames = os.listdir(self.location)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = self.filenames[idx]
        img = np.load(self.location+name)
        img = img.astype('float')

        #shape of output image: channels * x-pixels * y-pixels
        img = np.reshape(img,(1,200,200))

        nm = name.split("_")
        if nm[0]=="Pocked":
            label = 1
        elif nm[0]=="Unpocked":
            label = 0
        else:
            print("String Parsing Error")

        label = np.array(label).astype('float')
        label = np.reshape(label,(1))

        return torch.from_numpy(img).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

# Test
if __name__ == '__main__':
    test_dataset = Cell_dataset('test/')
    test_loader = DataLoader(test_dataset,batch_size=128, shuffle=True)
    for idx, data in enumerate(test_loader):
        if idx%500 == 0:
            images = data[0]
            labels = data[1]
            print(images.size()) #batch_size x 1 x 200 x 200
            print(labels.size()) #batch_size x 1

            #display images
            plt.imshow(images[idx,0,:,:])
            print(labels[idx,0]) #Outputs: 0 = Unpocked, 1 = Pocked
