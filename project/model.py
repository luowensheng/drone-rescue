#%%
import torch
import torchvision
import numpy as np
import cv2, os
from torchvision.transforms import transforms 

class KRCNN:
    def __init__(self):          

        # transform to convert the image to tensor
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # initialize the model
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
                                                                    num_keypoints=17)
        # set the computation device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load the model on to the computation device and set to eval mode
        self.model.to(self.device).eval()

    def predict(self, image_np):
        image = self.transform(image_np).unsqueeze(0).to(self.device)
        with torch.no_grad():
            
            outputs = self.model(image)
        return image_np, outputs    


    def draw_results(self, img, outputs, conf):

        scores = outputs[0]['scores']

        conf_indices = [ i for i in range(len(scores)) if scores[i]>conf]
        if not  any(conf_indices):
           return False     

        boxes = outputs[0]['boxes'][conf_indices].detach().cpu().numpy().astype(int)

        keypoints_scores = outputs[0]['keypoints_scores'][conf_indices].detach().cpu().numpy().astype(int)
        keypoints = outputs[0]['keypoints'][conf_indices].detach().cpu().numpy().astype(int)


        self.draw_boxes(img, boxes, thickness=1)  
        self.draw_keypoints(img, keypoints, keypoints_scores,color=(0,255,0), thickness=1 , radius=1 )
        return True
        
    
    def draw_boxes(self, img, boxes, color=(255,0,0), thickness=2):

        for (px1, py1, px2, py2 ) in boxes:
            start_point = (px1, py1)
            end_point= (px2, py2)
            cv2.rectangle(img, start_point, end_point, color=color, thickness=thickness)

    def draw_keypoints(self, img, keypoints, keypoints_scores , color=(0,255,0), thickness=2, radius=2, conf=0.5 ):
    
        for i, data in enumerate(keypoints):
        
            for j, (px, py, _) in enumerate(data):  
                if keypoints_scores[i][j]<conf:
                    continue       
                center = (px, py)
                cv2.circle(img, center, radius, color, thickness)

    def load_img(self, data):
        if type(data)==str:
           return cv2.imread(data) 
        if type(data)==np.ndarray:
           return data 
        return type(data)
            
    def __call__(self, data, min_confidence:float = 0.2) -> np.array : 
        image_np = self.load_img(data)
        img, outputs = self.predict(image_np)    
        contains_detection = self.draw_results(img, outputs, min_confidence)
        return img 


        
    
