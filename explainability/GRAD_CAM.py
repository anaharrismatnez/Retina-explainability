import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import logging
from torchvision import models
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import cv2



#GRAD-CAM IMPLEMENTATION FROM: ##################################################
# https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
#################################################################################

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


##############################
CLASSES = ['DR','no_DR']
#CLASSES = ['CACSmenos400','CACSmas400']
#directory = '/home/aharris/shared/EyePACS/input/image/dynamic_run/test'
directory = '/home/aharris/shared/indianDB/train_seg'
#out_path = '/home/aharris/shared/EyePACS/interpretability/GRAD_CAM'
out_path = '/home/aharris/shared/indianDB/GradCAM'
model_path = '/home/aharris/shared/EyePACS/models/exp10/weights_200.pth'
##############################

class MODEL(nn.Module):
    def __init__(self,base_model):
        super(MODEL, self).__init__()
        
        # get the pretrained VGG16 network
        self.MODEL = models.vgg16(pretrained=True)

        # Newly created modules have require_grad=True by default 
        num_features = self.MODEL.classifier[6].in_features
        features = list(self.MODEL.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, 2)

        features.extend([linear])  # Add our layer with 2 outputs
        self.MODEL.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Load pre initialized model
        self.MODEL.load_state_dict(torch.load(base_model))
        
        #summary(model, (1, 3, 224, 224))
        # disect the network to access its last convolutional layer
        self.features_conv = self.MODEL.features[:29]

        # get the relu and the max pool of the features stem
        #self.relu_max_pool_features = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.relu_max_pool_features = self.MODEL.features[29:31]
        
        # get the classifier of the vgg16
        self.classifier = self.MODEL.classifier
        
        # placeholder for the gradients
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features_conv(x)
        
        # register the hook
        x.register_hook(self.activations_hook)
        
        x = self.relu_max_pool_features(x)
        x = x.view((1, -1))
        #print(np.shape(x))
        x = self.classifier(x)
        return x
    
    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, x):
        return self.features_conv(x)

def overlay(directory,heatmap):
    img = cv2.imread(directory)
    heatmap = cv2.resize(np.float32(heatmap), (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap  + img * 0
    return superimposed_img

def img_to_tensor(root,file):
    img = Image.open(os.path.join(root,file))

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4387, 0.3090, 0.2211], std=[0.2733, 0.2035, 0.1717]),
        ])

    # unsqueeze provides the batch dimension
    img_tensor = tfms(img).unsqueeze(0)

    return img_tensor

if __name__ == '__main__':
    start_time = time.time()
    
    # Image datasets
    image_test_folder = directory

    assert os.path.exists(image_test_folder)

    # Load CNN Model
    device = torch.device('cpu')

    # Import model
    model = MODEL(model_path)
    model.to(device=device)
    

    if not os.path.exists(out_path):
        os.mkdir(out_path) 
    
    success = 0
    total = 0

    
    for root, dirs, files in os.walk(image_test_folder):
        path = root.split(os.sep)
        
        for file in files:
            
            total += 1
            #label = 0 if 'mas' in path[-1] else 1
            print (f'[{file}]:')
            
            if root == (f'{directory}/{CLASSES[0]}'):

                cd = f'{out_path}/{CLASSES[0]}'
                if not os.path.exists(cd):
                    os.mkdir(cd) 
                                
                img_tensor = img_to_tensor(root,file)

                model.eval()

                # get the most likely prediction of the model
                pred = model(img_tensor)
                
                pred[:,1].backward()

                # pull the gradients out of the model 
                #The size is dictated by the spacial dimensions of the activation maps in the last convolutional layer of the network.
                gradients = model.get_activations_gradient()
                
                # pool the gradients across the channels 
                # Returns the mean value of each row of the input tensor in the given dimension dim. 
                # If dim is a list of dimensions, reduce over all of them.
                #COMPUTING THE ALPHA VALUES
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
                
                # get the activations of the last convolutional layer
                activations = model.get_activations(img_tensor).detach()
                
                # weight the channels by corresponding gradients
                for i in range(512):
                    activations[:, i, :, :] *= pooled_gradients[i]
                    
                # average the channels of the activations
                heatmap = torch.mean(activations, dim=1).squeeze()

                # relu on top of the heatmap
                heatmap = np.maximum(heatmap, 0)

                # normalize the heatmap
                heatmap /= torch.max(heatmap)
                #print(heatmap.size)
                
                # draw the heatmap
                plt.matshow(heatmap.squeeze())
                plt.axis('off')
                plt.savefig('{}/{}'.format(cd,file), dpi=400,bbox_inches='tight',pad_inches=0)
                plt.clf()

                #overlay the heatmap
                img_directory = root+'/'+file
                superimposed_img = overlay(img_directory,heatmap)

        
                cv2.imwrite('{}/{}_map_smoothed.jpg'.format(cd,file), superimposed_img)

                

            if root == (f'{directory}/{CLASSES[1]}'):
                cd = f'{out_path}/{CLASSES[1]}'
                if not os.path.exists(cd):
                    os.mkdir(cd) 
                    
                img_tensor = img_to_tensor(root,file)
                
                model.eval()
                # get the most likely prediction of the model
                pred = model(img_tensor)
                pred[:,1].backward()

                # pull the gradients out of the model
                gradients = model.get_activations_gradient()
                

                # pool the gradients across the channels
                pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

                # get the activations of the last convolutional layer
                activations = model.get_activations(img_tensor).detach()
                
                
                # weight the channels by corresponding gradients
                for i in range(512):
                    activations[:, i, :, :] *= pooled_gradients[i]
                    
                # average the channels of the activations
                heatmap = torch.mean(activations, dim=1).squeeze()

                # relu on top of the heatmap
                heatmap = np.maximum(heatmap, 0)

                # normalize the heatmap
                heatmap /= torch.max(heatmap)

                # draw the heatmap
                plt.matshow(heatmap.squeeze())
                plt.savefig('{}/{}'.format(cd,file))
                plt.clf()

                #overlay the heatmap
                img_directory = root+'/'+file
                superimposed_img = overlay(img_directory,heatmap)
         
                cv2.imwrite('{}/{}_map_smoothed.jpg'.format(cd,file), superimposed_img)

                
                
    print("--- %s seconds ---" % (time.time() - start_time))