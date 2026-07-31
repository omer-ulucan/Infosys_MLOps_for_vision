'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import shutil
from PIL import Image, ImageDraw

import utility.constants as const

def crop_object_from_image(image_path,img, annotation,outputPath,parentFolder,saveCroppedImage,saveImageWithbbox):
    """
    Crop the image based on the given annotation.

    Parameters:
    - image_path: str, path to the image file
    - annotation: tuple, (x, y, width, height) representing the bounding box to crop

    Returns:
    - cropped_image: PIL.Image object, the cropped image
    """
    # Open the image file
    prefix = const.CROPPED_IMAGE_PREFIX
    
    # Extract the bounding box coordinates
    x, y, width, height = annotation
    # Calculate the right and bottom coordinates
    right = x + width
    bottom = y + height
    # Crop the image
    cropped_image = img.crop((x, y, right, bottom)).convert("RGB")
    #folder_path = os.path.dirname(image_path)
    
    if(saveCroppedImage):
        #create cropped images folder
        if os.path.exists(os.path.join(  outputPath + "/" + const.CROPPED_FOLDER_PREFIX)) == False:
           os.mkdir(os.path.join(  outputPath + "/" + const.CROPPED_FOLDER_PREFIX) )

        #create object/sub category folder
        saveCropPath =  os.path.join( outputPath + "/" + const.CROPPED_FOLDER_PREFIX +"/"+  parentFolder)
        if os.path.exists(saveCropPath):
            #shutil.rmtree(saveCropPath)
            pass
        else:
            os.mkdir(saveCropPath)
        
        target_path = os.path.join( saveCropPath+ "/"+ prefix + '_' + os.path.basename(image_path))
        cropped_image.save(target_path)
    if(saveImageWithbbox):
        #create bbox images folder
        if os.path.exists(os.path.join( outputPath + "/" + const.IMAGE_FOLDER_WITH_BBOX)) == False:
           os.mkdir(os.path.join( outputPath + "/" + const.IMAGE_FOLDER_WITH_BBOX) )

        #create object/sub category folder
        saveImgBboxCropPath = os.path.join(  outputPath+ "/" + const.IMAGE_FOLDER_WITH_BBOX +"/" +parentFolder )
        if os.path.exists(saveImgBboxCropPath):
            #shutil.rmtree(saveImgBboxCropPath)
            pass
        else:
            os.mkdir(saveImgBboxCropPath)
        
        draw = ImageDraw.Draw(img)
        draw.rectangle((x, y, right, bottom), outline="red", width=3)
        
        target_path = os.path.join( saveImgBboxCropPath + "/"+  prefix + '_' + os.path.basename(image_path))
        img.save(target_path)
    return cropped_image

def yolo_to_xywh(yolo_annotation, img_width, img_height):
    """
    Convert normalized YOLO annotation to (x, y, width, height) format.


    Parameters:
    - yolo_annotation: tuple, (class_id, x_center_norm, y_center_norm, width_norm, height_norm)
    - img_width: int, width of the image
    - img_height: int, height of the image

    Returns:
    - xywh_annotation: tuple, (class_id, x, y, width, height)
    """

    annList = yolo_annotation.split(' ')
    class_id = annList[0]
    x_center_norm = annList[1]
    y_center_norm = annList[2]
    width_norm = annList[3]
    height_norm = annList[4]

    # Convert normalized values to absolute values
    x_center = float(x_center_norm) * float(img_width)
    y_center = float(y_center_norm) * float(img_height)
    width = float(width_norm) * float(img_width)
    height = float(height_norm) * float(img_height)

    # Calculate top-left corner coordinates
    x = x_center - (width / 2)
    y = y_center - (height / 2)

    return (x, y, width, height)