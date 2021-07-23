from model import KRCNN
import time
import cv2
import os
import numpy as np
import  time
from tqdm import tqdm

# https://towardsdatascience.com/image-processing-and-pixel-manipulation-photo-filters-5d37a2f992fa


def get_array(shape, inc =150, size=200):
    inc = size
    i = 0
    x = 0
    x_not_done = True
    while x_not_done:
        if (i+size)< shape[0]:
            xs, xe = i,i+size
        else:
            xs, xe = i, shape[0]
            x_not_done = False
            
               
        j = 0
        y = 0
        y_not_done = True
        while y_not_done:
           if (j+size) > shape[1]:
              yield [x,y],[xs, xe], [j,shape[1]] 
              y_not_done = False
           else:
               yield [x,y],[xs, xe], [j,j+size] 
                  
           j+=inc
           y+=1
        i+=inc 
        x+=1   
        
def hist_eq(img):
    R, G, B = cv2.split(img)

    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)

    equ = cv2.merge((output1_R, output1_G, output1_B))            
    return equ    
   

IMG_DIR ="D:/Downloads/Compressed/VisDrone2019-MOT-train/VisDrone2019-MOT-train/sequences/uav0000013_00000_v"
#"D:/Downloads/Compressed/VisDrone2019-MOT-train/VisDrone2019-MOT-train/sequences/uav0000020_00406_v"
#IMG_DIR = "D:/Downloads/Compressed/VisDrone2019-DET-train/VisDrone2019-DET-train/images"
model = KRCNN()

images = os.listdir(IMG_DIR)

get_path = lambda images=images, img_dir=IMG_DIR : os.path.join(img_dir,  np.random.choice(images) )  # "0000001.jpg") 

path = get_path()

img = cv2.imread(path)



imgp = (img+0)*1

imgp = hist_eq(imgp)
imgp, _ = model(imgp, return_img=True)



quit()
#
size = 400
sizes = [400, 300] # 
sizes = [500,400, 350, 300, 200] #, 100, 50]
for ii in range(2):
    
    for size in tqdm(sizes):
        img = cv2.imread(path)
        i_x = 0
        i_y = 0
        keep_y = []
        img2 = []
        width = 5
        shape = (img.shape)
        
        startt = time.time()

        for i, ((x,y),(start_x, end_x),(start_y, end_y)) in enumerate(get_array(shape, size=size)):
            
            patch = img[start_x:end_x,start_y:end_y,:]
            if ii == 0:
                patch = hist_eq(patch)
            patch, contains_detection = model(patch, return_img=True)
            if not contains_detection:
                patch*=0
            
            patch[0:width,:,: ] = 255
            patch[shape[0]-width: shape[0],:,: ] = 255
            patch[:,0:width,: ] = 255
            patch[:,shape[1]-width:shape[1],: ] = 255
            
            #print(size, i, patch.shape)
            #print(f"i = {i}  x  = {x}  y = {y} i_y = {i_y} ==> {(start_x, end_x),(start_y, end_y)}  {start_x==i_x}")
            if start_x!=i_x:
                temp = np.concatenate(keep_y, axis=1)
                img2.append(temp) 
                keep_y = []
                i_x = start_x
                
            keep_y.append(patch)    


        img2 = np.concatenate(img2, axis=0)
        endt = np.round(time.time() - startt, 3 )          
        cv2.imwrite(os.path.join("outputs",f"equalizeHist_patches_{size}_{endt}_seconds.png"), img2)
    cv2.imwrite(os.path.join("outputs",f"equalizeHist_img_pred.png"), imgp)

    #cv2.waitKey(10)
    #cv2.destroyAllWindows()    
        


