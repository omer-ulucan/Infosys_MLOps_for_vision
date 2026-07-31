'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import shutil
import xlsxwriter
import cv2
import torch
import torchvision.ops.boxes as bops
from utility.json_ops import get_dict_from_json


def generate_detailed_report(report_path, images_path, annotation_path, predicted_annotation_path, metadata_path,
                             tag_list, inference_image_path):
    try:
        if os.path.exists(inference_image_path):
            shutil.rmtree(inference_image_path)
        else:
            pass
        os.mkdir(inference_image_path)
        image_row = 1
        # Creating Excel Workbook..
        workbook = xlsxwriter.Workbook(report_path + '/Detailed Report.xlsx')
        # Defining the Alignment required..
        my_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': 'True'})
        # Creating worksheet..
        worksheet = workbook.add_worksheet('Ground Truth Validation Report')
        worksheet.set_column(0, 0, 10)
        worksheet.set_column(1, 20, 50)
        worksheet.set_column(2, 2, 55)
        worksheet.set_default_row(266)
        worksheet.set_row(0, 14.5)
        # Defining Header Data..
        worksheet.write(0, 0, 'Sr.No', my_format)
        worksheet.write(0, 1, 'File Name', my_format)
        worksheet.write(0, 2, 'Image', my_format)
        worksheet.write(0, 3, 'Predicted Annotations', my_format)
        worksheet.write(0, 4, 'Original Annotations', my_format)
        worksheet.write(0, 5, 'IOU Threshold Annotations', my_format)
        dir_list = os.listdir(images_path)
        for image_data in dir_list:
            if image_data.endswith(('.png', '.jpg', '.jpeg')):
                img_read_val_detail = cv2.cvtColor(
                    cv2.imread(images_path + image_data), cv2.COLOR_BGR2RGB)
                dh, dw = img_read_val_detail.shape[:2]
                list_of_info_detail = []
                predicted_annotation_detail = []
                try:
                    file_txt_name = os.path.splitext(image_data)
                    if os.path.exists(predicted_annotation_path + file_txt_name[0] + '.txt'):
                        for line in open(predicted_annotation_path + file_txt_name[0] + '.txt'):
                            predicted_annotation_dict_detail = dict();
                            if line.rstrip('\n'):
                                label_pred, prob, left_pred, top_pred, width_pred, height_pred = [
                                    int(element) if index != 1 else float(element) for
                                    index, element in
                                    enumerate(line.rstrip('\n').split(' '))]

                                list_of_info_detail.append([left_pred, top_pred, width_pred, height_pred])
                                predicted_annotation_dict_detail['tag_name'] = tag_list[label_pred]
                                predicted_annotation_dict_detail['bounding_box.left'] = left_pred
                                predicted_annotation_dict_detail['bounding_box.top'] = top_pred
                                predicted_annotation_dict_detail['bounding_box.width'] = width_pred
                                predicted_annotation_dict_detail['bounding_box.height'] = height_pred
                                predicted_annotation_detail.append(predicted_annotation_dict_detail)
                        for predicted_result_detail in list_of_info_detail:
                            cv2.rectangle(img_read_val_detail,
                                          (predicted_result_detail[0], predicted_result_detail[1]),
                                          (predicted_result_detail[2], predicted_result_detail[3]),
                                          (0, 0, 0), 6)
                    else:
                        predicted_annotation_detail = []
                        pass
                except Exception as e:
                    print(e)

                list_of_info = []
                original_annotation = []
                try:
                    file_txt_name = os.path.splitext(image_data)
                    for line in open(annotation_path + file_txt_name[0] + '.txt'):
                        original_annotation_dict = dict()
                        if line.rstrip('\n'):
                            label, x, y, w, h = [float(element) if index > 0 else int(element) for
                                                 index, element in
                                                 enumerate(line.rstrip('\n').strip().split(' '))]
                            l = int(max((x - w / 2) * dw, 0))
                            r = int(min((x + w / 2) * dw, dw - 1))
                            t = int(max((y - h / 2) * dh, 0))
                            b = int(min((y + h / 2) * dh, dh - 1))
                            list_of_info.append([l, t, r, b])
                            original_annotation_dict['tag_name'] = tag_list[label]
                            original_annotation_dict['bounding_box.left'] = l
                            original_annotation_dict['bounding_box.top'] = t
                            original_annotation_dict['bounding_box.width'] = r
                            original_annotation_dict['bounding_box.height'] = b
                            original_annotation.append(original_annotation_dict)
                    for original_result in list_of_info:
                        cv2.rectangle(img_read_val_detail, (original_result[0], original_result[1]),
                                      (original_result[2], original_result[3]),
                                      (0, 255, 0), 6)

                except Exception as e:
                    print(e)

                try:
                    file_json_name = os.path.splitext(image_data)
                    meta_data_ground_truth = get_dict_from_json(
                        metadata_path + file_json_name[0] + '.json')
                    meta_data_ground_truth_data = meta_data_ground_truth.get('user')
                except Exception as e:
                    print(e)

                img = cv2.resize(img_read_val_detail, (350, 350))
                cv2.imwrite(inference_image_path + '/' + image_data, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                # Writing the Data in Excel Workbook
                worksheet.write(image_row, 0, image_row, my_format)
                worksheet.write(image_row, 1, image_data, my_format)

                worksheet.insert_image(image_row, 2, inference_image_path + '/' + image_data,
                                       {'align': 'center', 'valign': 'vcenter'})
                worksheet.write(image_row, 4, str(original_annotation), my_format)
                iou_data = []
                if len(predicted_annotation_detail) == 0:
                    worksheet.write(image_row, 3, 'No Prediction', my_format)
                    worksheet.write(image_row, 5, 'No Prediction', my_format)
                else:
                    worksheet.write(image_row, 3, str(predicted_annotation_detail), my_format)
                    for annotation in predicted_annotation_detail:
                        tag_val = annotation['tag_name']
                        original_list = []
                        for original_data in original_annotation:
                            if original_data['tag_name'] == tag_val:
                                original_list.append(original_data)
                            else:
                                pass
                        if len(original_list) == 0:
                            pass
                        else:
                            iou_val_prediction = []
                            for original_data_val in original_list:
                                annotation_dict_pred = annotation
                                annotation_dict_original = original_data_val
                                annotation_box_pred = [annotation_dict_pred['bounding_box.left'],
                                                       annotation_dict_pred['bounding_box.top'],
                                                       annotation_dict_pred['bounding_box.width'],
                                                       annotation_dict_pred['bounding_box.height']
                                                       ]
                                annotation_box_original = [annotation_dict_original['bounding_box.left'],
                                                           annotation_dict_original['bounding_box.top'],
                                                           annotation_dict_original['bounding_box.width'],
                                                           annotation_dict_original['bounding_box.height']
                                                           ]
                                box1 = torch.tensor([annotation_box_pred], dtype=torch.float)
                                box2 = torch.tensor([annotation_box_original], dtype=torch.float)
                                iou = bops.box_iou(box1, box2)
                                iou_percent = (iou * 100)
                                iou_val_prediction.append(int(iou_percent))
                            iou_data.append(max(iou_val_prediction))
                worksheet.write(image_row, 5, str(iou_data), my_format)
                col_val = 6
                for meta_data_key, meta_data_val in meta_data_ground_truth_data.items():
                    worksheet.write(0, col_val, meta_data_key, my_format)
                    worksheet.write(image_row, col_val, meta_data_val, my_format)
                    col_val += 1
                image_row += 1
            else:
                pass
        workbook.close()
    except Exception as e:
        print(e)
