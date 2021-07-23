import time
import cv2
import os
import numpy as np
import  time
from tqdm import tqdm

print(os.getcwd())

        
sequence = "uav0000013_00000_v"   

IMG_DIR = f"/krcnn/datastets/visdrone/sequences/{sequence}"
annotations_path = f"/krcnn/datastets/visdrone/annotations/{sequence}.txt"
images = os.listdir(IMG_DIR)

annotations = open(annotations_path).read().split('\n')



not_complete = True
color = (0,0,255)
thickness = 10
while not_complete:
    num = np.random.randint(len(images))
    img_path = os.path.join(IMG_DIR, images[num] ) 
    img = cv2.imread(img_path)
    tried = False
    not_complete=False
    for x in annotations:
        frame_num, seq_num, x1,x2,x3,x4,*_ = x.split(",")
        if int(seq_num)!=num:
            if tried: 
               not_complete = False
            break
        
        start_x, end_x, start_y, end_y = int(x1), int(x1)+int(x3), int(x2), int(x2)+int(x4)
        start_point, end_point = (int(x1), int(x1)+int(x3)), (int(x2), int(x2)+int(x4))
        cv2.rectangle(img, start_point, end_point, color=color, thickness=thickness)
        tried = True
        
cv2.imshow("IMAGE", img)
cv2.waitKey(1000)
cv2.destroyAllWindows()    
    

