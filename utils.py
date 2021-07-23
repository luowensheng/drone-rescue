from tqdm import tqdm
import numpy as np
import os, cv2

def get_patch_indices(shape, patchsize, stride=None):

    stride = patchsize if stride is None else stride
    start_vertical = 0
    has_not_reached_length = True
    img_width, img_height = shape

    while has_not_reached_length:

        if ( start_vertical+patchsize < img_width ):
            end_vertical = start_vertical + patchsize    
        else:
            end_vertical = img_width
            
    
        start_horizontal = 0
        has_not_reach_width = True

        while has_not_reach_width:

           if (start_horizontal+patchsize > img_height):
               end_horizontal  = img_height
               has_not_reach_width = False
           else:
               end_horizontal = start_horizontal+patchsize
            
           yield [start_vertical, end_vertical], [start_horizontal, end_horizontal] 
       
           start_horizontal+=stride
           
        if not ( vertical_index + patchsize < img_width ):       
            has_not_reached_length = False         
        vertical_index +=stride 
    
        
def run_on_patches(model,img, patchsize=400, add_separation_lines = True, width = 5) : 

        current_vertical_index = 0
        horizontal_patches = []
        output_img = []
        
        img_width, img_height = (img.shape)

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

        output_img.append(np.concatenate(horizontal_patches, axis=1)) 
        output_img = np.concatenate(output_img, axis=0)
        return output_img
    
