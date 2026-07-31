'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import json
# import pybboxes as pbx
# import torch
# import torchvision.ops.boxes as bops
from pathlib import Path
import os




def get_dict_from_json(path):
    """ JSON TO DICT conversion """
    try:
     
        config_path =  os.getcwd() + "/" + path

        with open(config_path, 'r') as f:
            output = json.load(f)
            return output
    
    except Exception as e:
        print(e)


def create_json(path, filename, data):
    file_filename = path + '/' + filename + '.json'
    final_data = {"user": data}
    with open(file_filename, 'w') as fp:
        json.dump(final_data, fp)


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

