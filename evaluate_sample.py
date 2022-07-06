
import torch
from utils.paths import *
from PIL import Image
import torchvision
from torch.utils.data import DataLoader
from utils.cnn import load_model
import matplotlib.pyplot as plt
import seaborn as sn
import os
import numpy as np
import pandas as pd


# Image datasets
image_test_folder = '/home/aharris/shared/CAC_Assesment_NORM/crop_folds/fold_4'
assert os.path.exists(image_test_folder)

# Load CNN Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Import model
model = load_model('/home/aharris/shared/EyePACS/models/exp10/weights_200.pth')
model.to(device)

tfms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
test_set = torchvision.datasets.ImageFolder(image_test_folder, transform=tfms)
test_dataloader = DataLoader(test_set, batch_size= 1, pin_memory = True)

test_correct = 0
test_total = 0
TP = 0
FN = 0
FP = 0
TN = 0
model.eval()
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        sample_fname, _ = test_dataloader.dataset.samples[i]
        print(sample_fname)
        sample, ground = data
        sample = sample.cuda()
        ground = ground.cuda()
        #print(sample)

        outputs = model(sample)
        #print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += ground.size(0)
        test_correct += (predicted == ground).sum().item()

        if ground == 1 and predicted == 0:
            FN += 1
        if ground == 1 and predicted == 1:
            TP += 1

        if ground == 0 and predicted == 0:
            TN += 1
        if ground == 0 and predicted == 1:
            FP +=1


        print(f'Label: {ground.data.cpu().numpy()} --> Prediction: {predicted.item()}'
                f'{"--> OK" if predicted.item() == ground else ""}')

""" classes = ('DR', 'no_DR')
total_positive = len(os.listdir(f'{image_test_folder}/{classes[1]}'))
total_negative = len(os.listdir(f'{image_test_folder}/{classes[0]}'))

matrix = np.empty([2,2])
matrix[0,0] =(TN/total_negative)*100 
matrix[0,1] =(FP/total_negative)*100 
matrix[1,0] =(FN/total_positive)*100 
matrix[1,1] =(TP/total_positive)*100 

df_cm = pd.DataFrame(matrix, index = [i for i in classes],
                    columns = [i for i in classes])

plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig(f'confusionmatrix_exp10_200_IDRiD.png') """

print(f'-----------------------\nModel accuracy: {(test_correct * 100) / test_total:.2f}')
