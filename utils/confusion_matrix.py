import pandas as pd
import numpy as np
import seaborn as sn
from cnn import load_model
import torchvision
import time
from torch.utils.data import DataLoader
import torch
import sys
import os
import matplotlib.pyplot as plt

CLASSES = ['CACSmas400', 'CACSmenos400']

test_dir = '/home/aharris/shared/CAC_Assesment_NORM/input/image/dynamic_run/test'

model=load_model('/home/aharris/shared/EyePACS/models/exp11/weights_59.pth')

#model.to(device)
model.cuda()

tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4387, 0.3090, 0.2211], std=[0.2733, 0.2035, 0.1717]),
        ])

t0 = time.time()

test_set = torchvision.datasets.ImageFolder(test_dir, transform=tfms)

test_dataloader = DataLoader(test_set, batch_size=1)

len_positive = len(os.listdir(f'{test_dir}/{CLASSES[1]}'))
len_negative = len(os.listdir(f'{test_dir}/{CLASSES[0]}'))

TP = 0
FN=0
FP=0
TN=0
test_correct = 0
test_total = 0
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

        if ground == 0:
            if predicted == 1:
                FP += 1
            else:
                TN += 1
        else:
            if predicted == 1:
                TP += 1
            else:
                FN += 1


        print(f'Label: {ground.data.cpu().numpy()} --> Prediction: {predicted.item()}'
                f'{"--> OK" if predicted.item() == ground else ""}')


print('Test accuracy:',100*(test_correct/test_total))

FP = (FP/len_negative) * 100
TN = (TN/len_negative) * 100
TP = (TP/len_positive) * 100
FN = (FN/len_positive) * 100

matrix = np.empty((2,2))
matrix[0][0] = TN
matrix[0][1] = FP
matrix[1][0] = FN
matrix[1][1] = TP

df_cm = pd.DataFrame(matrix, index = [i for i in CLASSES],
                        columns = [i for i in CLASSES])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig(f'confusionmatrix_CAC.png')