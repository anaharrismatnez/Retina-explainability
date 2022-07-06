"""
01/12/2021

Functions to deal with the cnn operations
"""
import os
import numpy as np
import logging
from torchvision import models
import torch
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
from torch import optim
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
import torchvision

#CLASSES = ['DR','no_DR']
CLASSES = ['CACSmenos400','CACSmas400']

def load_model(base_model):
    '''
       Gets VGG16 model
       :param base_model: path to pre initialized model
       :return: vgg16 model with last layer modification (2 classes)
       '''
    model = torchvision.models.vgg16(pretrained=True)

    # Freeze trained weights
    for param in model.features.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    linear = torch.nn.Linear(num_features, 2)

    features.extend([linear])  # Add our layer with 2 outputs
    model.classifier = torch.nn.Sequential(*features)  # Replace the model classifier

    model.load_state_dict(torch.load(base_model))
   
    return model

def get_model():
    '''
       Gets VGG16 model
       :param base_model: path to pre initialized model
       :return: vgg16 model with last layer modification (2 classes)
       '''
    model = models.vgg16(pretrained=True)

    # Freeze trained weights
    for param in model.features.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    linear = nn.Linear(num_features, 2)

    features.extend([linear])  # Add our layer with 2 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    
    logging.info(f'Loading pretrained VGG16 model')

    nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')

    return model


def train_eval_model(writer, model, device, train_loader,val_loader, MODELS_FOLDER,batch_size=4, lr=0.1,epochs=1):

    


    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    last_loss = 100
    patience = 2
    trigger_times = 0
    
    
    
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.96
            ) 

    #scheduler = optim.ExponentialLR(optimizer, gamma=0.96)
    best_model = -1
    best_acc = 0.0
    model.train(True)
    for epoch in range(epochs):
        correct = 0
        total = 0
        running_loss = 0
        valid_loss = 0.0
        correct_val = 0
        total_val = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i,batch in enumerate(train_loader):
                sample, ground = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = prediction.max(1)

                total += ground.size(0)
                correct += (predicted == ground).sum().item()
                train_accuracy = (100 * correct) / total
                train_loss = running_loss/n_train
                pbar.set_postfix(**{'Train loss': train_loss, 'Train accuracy':train_accuracy})
                pbar.update(sample.shape[0])

        with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            model.eval() 
            for i,batch in enumerate(val_loader):    
                sample, ground = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)        

                #forward pass
                prediction = model(sample)
                loss = criterion(prediction, ground)
                running_loss += loss.item()

                _, predicted = prediction.max(1)

                total_val += ground.size(0)
                correct_val += (predicted == ground).sum().item()
                val_accuracy = (100 * correct_val) / total_val
                val_loss = running_loss/n_val

                pbar.set_postfix(**{'Val loss': val_loss, 'Val accuracy':val_accuracy})
                pbar.update(sample.shape[0])

            lr_scheduler.step()

            """ if val_accuracy > best_acc + 0.005:
                best_model = epoch
                best_acc = val_accuracy
                torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_final.pth")
                ) """

            if epoch % 10 == 0:
                torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_{epoch}.pth")
                )

            if epoch == (epochs -1):
                torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_{epoch}.pth")
                )

            """ # Early Stopping
            if val_loss > last_loss:
                trigger_times += 1
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    break

            else:
                print('trigger times: 0')
                trigger_times = 0 """

            last_loss = val_loss

            
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.close()

    return model, train_loss, train_accuracy, val_loss, val_accuracy


def evaluate_model(model, dataloader, device,path):
    correct = 0
    total = 0
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            sample, ground = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

            y_pred.extend(predicted.data.cpu().numpy())
            y_true.extend(ground.data.cpu().numpy())


    # constant for classes
    classes = ('DR', 'no_DR')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *100, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'{path}/confusionmatrix.png')

    return (100 * correct) / total