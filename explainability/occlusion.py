""" 
Occlusion maps: Each pixel corresponds to the probability of the prediction with that pixel occluded respect the ground_truth 

"""
import torch
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from PIL import Image
from torchvision import transforms, models
import pickle
import sys
import gc


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

##############################
CLASSES = ['DR','no_DR']
#CLASSES = ['CACSmenos400','CACSmas400']
#directory = '/home/aharris/shared/EyePACS/input/image/dynamic_run/test'
directory = '/home/aharris/shared/indianDB/train_seg'
#out_path = '/home/aharris/shared/EyePACS/interpretability/GRAD_CAM'
out_path = '/home/aharris/shared/indianDB/occlusion'
model_path = '/home/aharris/shared/EyePACS/models/exp10/weights_200.pth'
##############################



#https://stackoverflow.com/questions/33282368/plotting-a-2d-heatmap-with-matplotlib
#https://blog.quantinsti.com/creating-heatmap-using-python-seaborn/
#https://www.kaggle.com/fqpang/visualizing-cnn-using-pytorch
#https://github.com/zhoubolei/CAM
#https://www.analyticsvidhya.com/blog/2018/03/essentials-of-deep-learning-visualizing-convolutional-neural-networks/



def evaluate_model(image,model):
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    probs = torch.softmax(output.data, 1)
    predicted,idx = torch.max(probs.data, dim=1)
    return probs.cpu().detach().numpy(),idx.cpu().detach().numpy()



def image_occlusion(image, label,model,device, path, occ_size = 50, occ_stride = 50, occ_pixel = 0.5):
  
    #get the width and height of the image
    width, height = image.shape[-2], image.shape[-1]

    #setting the output image width and height (default 16 occluded images)
    output_height = int(np.ceil((height-occ_size)/occ_stride))
    output_width = int(np.ceil((width-occ_size)/occ_stride))

    #print(width,height,output_width,output_height)

    #create a white image of sizes we defined
    heatmap = np.zeros((output_height, output_width))
    #print(height,width)
    
    COUNT = 0
    #iterate all the pixels in each column
    predictions = []
    for h in range(0, height):
        for w in range(0, width):
            
            h_start = h*occ_stride
            w_start = w*occ_stride
            h_end = min(height, h_start + occ_size)
            w_end = min(width, w_start + occ_size)
            
            if (w_end) >= width or (h_end) >= height:
                continue
            
            input_image = image.clone().detach()
            #replacing all the pixel information in the image with occ_pixel(grey) in the specified location
            input_image[:,:, w_start:w_end, h_start:h_end] = occ_pixel
            #show(input_image)
            input_image = input_image.to(device=device)
            #run inference on modified image
            probs,idx = evaluate_model(input_image,model)


            if label == 0:

                prob_occ = probs[0][0]
              
                

            if label == 1:

                prob_occ = probs[0][1]
               
                
            COUNT = COUNT + 1

            heatmap[w,h] = prob_occ
            
            
    #print(np.shape(heatmap))

    
    return heatmap, predictions

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

def load_model(base_model):
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
    linear = torch.nn.Linear(num_features, 2)

    features.extend([linear])  # Add our layer with 2 outputs
    model.classifier = torch.nn.Sequential(*features)  # Replace the model classifier

    model.load_state_dict(torch.load(base_model))
   
    return model

if __name__ == '__main__':
    start_time = time.time()
    
    # Image datasets
    image_test_folder = directory

   
    assert os.path.exists(image_test_folder)

    # Load CNN Model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Import model
    model = load_model(model_path)
    model.to(device=device)

    if not os.path.exists(out_path):
        os.mkdir(out_path)

    success = 0
    total = 0
        
    for root, dirs, files in os.walk(image_test_folder):
        path = root.split(os.sep)

        for file in files:
            total += 1

            print (f'[{file}]:')

            if root == (f'{directory}/{CLASSES[0]}'):
                
                cd = f'{out_path}/{CLASSES[0]}'
                if not os.path.exists(cd):
                    os.mkdir(cd)
                label = 0                               
                img_tensor = img_to_tensor(root,file)
                
                model.eval()
                heatmap,predictions = image_occlusion(img_tensor,label,model,device,path,4,4)


                #print('Score without occlusion:',prob_no_occ)
        
                heatmap_img = plt.imshow(heatmap, interpolation='bicubic')
                heatmap_img.set_cmap('jet')
                plt.axis('off')
                plt.savefig(f'{cd}/{file}.png', dpi=400,bbox_inches='tight',pad_inches=0)
                fig,ax = plt.subplots()
                plt.colorbar(heatmap_img,ax=ax)
                ax.remove()
                plt.savefig(f'{cd}/{file}_colorbar.png'.format(file))
                plt.figure().clear()

                
                
                
        
            if root == (f'{directory}/{CLASSES[1]}'):
                label = 1
                cd = f'{out_path}/{CLASSES[1]}'
                if not os.path.exists(cd):
                    os.mkdir(cd)
                img_tensor = img_to_tensor(root,file)
                
                model.eval()
                heatmap,predictions = image_occlusion(img_tensor,label,model,device,path,4,4)


                #print('Score without occlusion:',prob_no_occ)
        
                heatmap_img = plt.imshow(heatmap, interpolation='bicubic')
                heatmap_img.set_cmap('jet')
                plt.axis('off')
                plt.savefig(f'{cd}/{file}.png', dpi=400,bbox_inches='tight',pad_inches=0)
                fig,ax = plt.subplots()
                plt.colorbar(heatmap_img,ax=ax)
                ax.remove()
                plt.savefig(f'{cd}/{file}_colorbar.png'.format(file))
                plt.figure().clear()

                
        
            

print("--- %s seconds ---" % (time.time() - start_time))


                
            



            
