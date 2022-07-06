from PIL import Image
import os
import time
import numpy as np
import matplotlib.pyplot as plt


start_time = time.time()


out_directory = '/home/aharris/shared/indianDB/occlusion/D'#,'/home/aharris/shared/EyePACS/interpretability/occlusion/no_DR']
directory = '/home/aharris/shared/indianDB/train_seg/'
extension = 'jpg'


for folder in os.listdir(directory):
    images = []
    path = os.path.join(directory,folder)
    out_path = os.path.join(out_directory,folder) 

    for file in os.listdir(path):
            if file.endswith(extension):
                img = file.split('.')
                images.append(img[0])

    for i in range(len(images)):
        print('--',images[i])

        background = Image.open('{}/{}.{}'.format(path,images[i],extension))
        background_dim = np.shape(background)
        background = background.convert("RGBA")  


        front = Image.open('{}/{}.{}.png'.format(out_path,images[i],extension))
        front = front.resize((background_dim[1],background_dim[0]))
        front = front.convert("RGBA") 

        overlay = Image.blend(background,front,0.8) 

        #front.save('{}/{}_map_smoothed.png'.format(out_path,images[i]))
        overlay.save('{}/{}_overlay.png'.format(out_path,images[i]))
        

        
          
      

print("--- %s seconds ---" % (time.time() - start_time))

