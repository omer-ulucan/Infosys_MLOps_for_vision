'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import pandas as pd
import PIL.Image
import cv2
import os
from utility import image_processor

def construct_next_best_match_string(data):
    result_string = ""

    for item in data:
        result_string += f"{item['class_name']} : {item['similarity_score']}\n"
    return result_string

def get_scale(image_path):
    img = PIL.Image.open(image_path)
    original_width,original_height = img.size
    max_width = 90  
    max_height = 90 
    if original_width > max_width:
        scaled_width = max_width
    else:
        scaled_width = original_width
    if original_height > max_height:
        scaled_height = max_height
    else:
        scaled_height = original_height

    x_scale = scaled_width / original_width
    y_scale = scaled_height / original_height
    return x_scale,y_scale

def draw_bounding_boxes(image_path, bbox,color, output_folder,is_yolo=False):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            return
        img_h , img_w ,_ = img.shape
        if is_yolo == True:
            #bbox = ' '.join(map(str, bbox))
            bbox = image_processor. yolo_to_xywh_withoutclassid(bbox, img_w, img_h)
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        right = x+w
        bottom= y+h
        cv2.rectangle(img, (x, y), (right, bottom), color, 10)  # Green rectangle
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # Save the image with bounding boxes
        image_filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_filename)
        cv2.imwrite(output_path, img)
        return output_path

def yolo_to_pascal_voc(x_center, y_center, w, h, img_w, img_h):
    x_center = float(x_center)
    y_center = float(y_center)
    w = float(w)
    h = float(h)
    w = w * img_w
    h = h * img_h
    x1 = (round(((2 * x_center * img_w) - w)/2))
    y1 = (round(((2 * y_center * img_h) - h)/2))
    x2 = (round(x1 + w))
    y2 = (round(y1 + h))
    pascal_voc_val = [x1, y1, x2, y2]
    pascal_voc_val_final = [0 if i < 0 else i for i in pascal_voc_val]
    return pascal_voc_val_final


def write_data_to_excel(data, excel_file_path,row_limit):
    formatted_data = construct_data(data)
    df = pd.DataFrame(formatted_data)
    workbook_index = 1
    start_row = 0
    current_workbook_path = f"{os.path.splitext(excel_file_path)[0]}_{workbook_index}.xlsx"
    writer = pd.ExcelWriter(current_workbook_path, engine='xlsxwriter')
    current_sheet = writer.book.add_worksheet('detailed_analysis')
    headers = df.columns.tolist()
    # Write headers to the first sheet of the first workbook
    for col_num, value in enumerate(headers):
        current_sheet.write(0, col_num, value)
    for index, item in enumerate(data):
        row_num = index - start_row + 1
        if row_num > row_limit:
            current_sheet.autofit()
            writer.close()  # Close the current workbook
            workbook_index += 1
            start_row = index
            current_workbook_path = f"{os.path.splitext(excel_file_path)[0]}_{workbook_index}.xlsx"
            writer = pd.ExcelWriter(current_workbook_path, engine='xlsxwriter')
            current_sheet = writer.book.add_worksheet('detailed_analysis')
            # Write headers to the new workbook's sheet
            for col_num, value in enumerate(headers):
                current_sheet.write(0, col_num, value)
            row_num = 1
        # Write data to the current sheet
        for col_num, value in enumerate(df.iloc[index].tolist()):
        #     current_sheet.write(row_num, col_num, value)
            if col_num == len(df.columns) - 1 and value.endswith(".html"):                 
                current_sheet.write_url(row_num, col_num, value, string=value)    
            else:                
                current_sheet.write(row_num, col_num, value) 
        gt_image_path = item['file_path']
        pred_image_path = item['predictions']['best_match']['file_path']
        output_folder = 'tmp'
        gt_bb_img = draw_bounding_boxes(gt_image_path, item['GT']['bounding_box'], (0, 255, 0), output_folder)
        pred_bb_img = draw_bounding_boxes(pred_image_path, item['predictions']['best_match']['bounding_box'], (0, 0, 255), output_folder)
        try:
            x_scale, y_scale = get_scale(gt_bb_img)
            current_sheet.insert_image(row_num, 5, gt_bb_img, {'x_scale': x_scale, 'y_scale': y_scale})
            x_scale, y_scale = get_scale(pred_bb_img)
            current_sheet.insert_image(row_num, 6, pred_bb_img, {'x_scale': x_scale, 'y_scale': y_scale})
        except Exception as e:
            print(f"Image not found: {e}")
    current_sheet.autofit()
    writer.close()

def construct_data(data):
    # Normalize nested dictionaries (GT and predictions)
    formatted_data = []
    for item in data:
        formatted_item={
            'Image name' : item['image_name'],
            "Category": item['category'],
            "Product":item['product'],
            # "Lighting":item['lighting'],
            # "Distance": item['distance'],
            #"angle": item["angle"],
            'Ground truth Image' : '',
            'Matched Image' : '',
            'Ground Truth Class': item['GT']['class_name'],
            'Predicted Class' : item['predictions']['best_match']['class_name'],
            'similarity score': item['predictions']['best_match']['similarity_score'],
            'next best match' : construct_next_best_match_string(item['predictions']['next_best_match']),
            'GT image path' : item['file_path'],
            'Matched image path' : item['predictions']['best_match']['file_path'],
            'Prediction result': item['predictions']['prediction_result'],
            'Ground Truth bbox': str(item['GT']['bounding_box']),
            'Predicted bbox': str(item['predictions']['best_match']['bounding_box']),
           
        }
        if 'plot_path' in item:
            formatted_item['3D plot'] = item['plot_path']
        formatted_data.append(formatted_item)
    return formatted_data


