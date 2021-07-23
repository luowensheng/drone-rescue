from tqdm import tqdm
import numpy as np
import os, cv2

def get_patch_indices(shape: tuple, patchsize:int, stride:int=None) -> list :
    """
    given a shape of (Height, Width), this function outputs indices of size (patchsize X patchsize)

    """
    stride = patchsize if stride is None else stride
    start_vertical = 0
    has_not_reached_length = True
    img_height, img_width = shape

    while has_not_reached_length:

        has_not_reached_length = start_vertical + patchsize < img_height 
        end_vertical = (start_vertical + patchsize) if has_not_reached_length else img_height 

        start_horizontal = 0
        has_not_reach_width = True

        while has_not_reach_width:

           has_not_reach_width =  start_horizontal+patchsize < img_width

           end_horizontal  = (start_horizontal + patchsize) if has_not_reach_width else img_width 
            
           yield [start_vertical, end_vertical], [start_horizontal, end_horizontal] 
       
           start_horizontal += stride


        start_vertical += stride 
    
        
def run_on_patches(model, img: np.array, patchsize:int, add_separation_lines:bool = True, width:int = 5)  -> np.array:
        """
        Given an image, patches of size (patchsize * patchsize * 3) are used for prediction 
        then reassambled as a single image
        """
        current_vertical_index = 0
        horizontal_patches = []
        output_img = []
        
        img_width, img_height, _ = img.shape

        for ((start_vertical, end_vertical),(start_horizontal, end_horizontal)) in get_patch_indices((img_width, img_height), patchsize=patchsize):
           
            started_new_vertical_row = start_vertical!=current_vertical_index

            # get patch from indices
            patch = img[start_vertical:end_vertical,start_horizontal:end_horizontal,:]

            # get prediction for current patch
            patch = model(patch)
        
            if add_separation_lines:
                patch[0:width,:,: ] = 255
                patch[img_width-width: img_width,:,: ] = 255
                patch[:,0:width,: ] = 255
                patch[:,img_height-width:img_height,: ] = 255
            
            # Append patches vertically    
            if started_new_vertical_row:
                output_img.append(np.concatenate(horizontal_patches, axis=1)) 
                horizontal_patches = []
                current_vertical_index = start_vertical

            # Append patch horizontally    
            horizontal_patches.append(patch)   
            #print(img.shape, patch.shape, [(start_vertical, end_vertical),(start_horizontal, end_horizontal)]) 
        
        # append final patches
        output_img.append(np.concatenate(horizontal_patches, axis=1))
        
        output_img = np.concatenate(output_img, axis=0)
        return output_img
    
