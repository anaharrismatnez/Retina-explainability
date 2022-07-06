import logging
import time
import torch
from torch.utils.data import DataLoader
from utils.cnn import get_model,load_model
import os
from torch.utils.tensorboard import SummaryWriter
import torchvision
from tqdm import tqdm
import numpy as np

import argparse

from torch import optim
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

""" train_dir ='/home/aharris/shared/EyePACS/input/image/dynamic_run/train'
test_dir = '/home/aharris/shared/EyePACS/input/image/dynamic_run/test'
val_dir = '/home/aharris/shared/EyePACS/input/image/dynamic_run/val' """

#Fine-tune CAC
train_dir ='/home/aharris/shared/CAC_Assesment_NORM/input/image/dynamic_run/train'
test_dir = '/home/aharris/shared/CAC_Assesment_NORM/input/image/dynamic_run/test'


######################################
# LOCAL HYPER-PARAMETERS             #
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 0.0001
TRAIN_STEP = False
######################################


parser = argparse.ArgumentParser(description='EyePACS training, select experiment.')
parser.add_argument('exp',  type=int,
                    help='Number of experiment')
args = parser.parse_args()

MODELS_FOLDER = f'/home/aharris/shared/EyePACS/models/exp{args.exp}'
if os.path.exists(MODELS_FOLDER) == False:
    os.mkdir(MODELS_FOLDER)

if __name__ == '__main__':
    # Check input consistence done. Check  /utils/check_input.py script for more information.
    writer = SummaryWriter()

    # log configuration
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] (%(asctime)s) : %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # Get model.
    #model = get_model()

    #Fine-tune CAC
    base_model = ''
    model=load_model('EyePACS/models/exp10/weights_200.pth')

    #model.to(device)
    model.cuda()

    tfms_t = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4387, 0.3090, 0.2211], std=[0.2733, 0.2035, 0.1717]),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        ])

    tfms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4387, 0.3090, 0.2211], std=[0.2733, 0.2035, 0.1717]),
        ])

    t0 = time.time()
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=tfms_t)
    #val_set = torchvision.datasets.ImageFolder(val_dir, transform=tfms)
    test_set = torchvision.datasets.ImageFolder(test_dir, transform=tfms)
    train_dataloader = DataLoader(train_set, batch_size= BATCH_SIZE, pin_memory = True, shuffle=True)
    #val_dataloader = DataLoader(val_set, batch_size= BATCH_SIZE, pin_memory = True, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size= 1, pin_memory = True)

    logging.info("Training model")
    
    logging.info(f'''Starting training:
        Epochs:          {EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Training size:   {len(train_dataloader.dataset)}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    criterion = torch.nn.CrossEntropyLoss()

    lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=5,
                gamma=0.96
            ) 

    last_epochs = np.arange(start=EPOCHS-5,stop=EPOCHS-1,step=1)
    for epoch in range(EPOCHS):
        correct = 0
        total = 0
        running_loss = 0
        
        with tqdm(total=len(train_dataloader.dataset), desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i,batch in enumerate(train_dataloader):
                model.train(True)
                sample, ground = batch
                sample = sample.cuda()
                ground = ground.cuda()

                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = prediction.max(1)

                total += ground.size(0)
                correct += (predicted == ground).sum().item()
                train_accuracy = 100 * (correct/ total)
                train_loss = running_loss/total
                pbar.set_postfix(**{'Train loss': train_loss, 'Train accuracy':train_accuracy})
                pbar.update(sample.shape[0])
               
            del(sample,ground,running_loss,predicted)
            lr_scheduler.step()

        logging.info("Evaluating model")
        running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for i,batch in enumerate(test_dataloader):    
                model.eval() 
                sample, ground = batch
                sample = sample.cuda()
                ground = ground.cuda()      

                #forward pass
                prediction = model(sample)
                loss = criterion(prediction, ground)
                running_loss += loss.item()

                _, predicted = prediction.max(1)

                total_val += ground.size(0)
                correct_val += (predicted == ground).sum().item()
                val_accuracy = 100 * (correct_val / total_val)
                val_loss = running_loss/ total_val
            
        del(sample,ground,predicted)
        print('Val loss:',val_loss, 'Val accuracy:',val_accuracy) 


        if epoch % 10 == 0:
                torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_{epoch}.pth")
                ) 

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.close()   

        
        if epoch in last_epochs:
                torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_{epoch}.pth")
                ) 

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.close()   

    logging.info("Testing model")
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


            print(f'Label: {ground.data.cpu().numpy()} --> Prediction: {predicted.item()}'
                    f'{"--> OK" if predicted.item() == ground else ""}')

    del(sample,ground)
    print('Test accuracy:',100*(test_correct/test_total))


    torch.save(
                    model.state_dict(),
                    str(f"{MODELS_FOLDER}/weights_{epoch}.pth")
                ) 


    


