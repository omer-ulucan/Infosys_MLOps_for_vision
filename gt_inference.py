'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os
import sys
import src.utils.converter as converter
import src.utils.general_utils as general_utils
from src.evaluators.pascal_voc_evaluator import (get_pascalvoc_metrics,
                                                 plot_precision_recall_curve,
                                                 plot_precision_recall_curves)
from src.utils.enumerators import BBFormat, BBType, CoordinatesType
import pandas as pd
import json
import shutil
import itertools
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def load_basic_operation(new_folder_gt):
    try:
        if os.path.exists(f'{new_folder_gt}/output'):
            shutil.rmtree(f'{new_folder_gt}/output')
        else:
            pass
        os.mkdir(f'{new_folder_gt}/output')
    except Exception as e:
        print(e)

    report_dir = f"{new_folder_gt}/reports"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)


def load_annotations_gt(new_folder_gt, dataset_name):
    ret = []
    ret = converter.yolo2bb(f"{new_folder_gt}/original Annotation",
                            f"{new_folder_gt}/original images",
                            f"{new_folder_gt}/{dataset_name}.names",
                            bb_type=BBType.GROUND_TRUTH)
    # Make all types as GT
    [bb.set_bb_type(BBType.GROUND_TRUTH) for bb in ret]
    return ret


def load_annotations_det(new_folder_gt, dataset_name):
    ret = []
    ret = converter.text2bb(f"{new_folder_gt}/predicted",
                            bb_type=BBType.DETECTED,
                            bb_format=BBFormat.XYX2Y2,
                            type_coordinates=CoordinatesType.ABSOLUTE,
                            img_dir=f"{new_folder_gt}/original images")
    # Verify if for the selected format, detections were found
    if len(ret) == 0:
        print('No file was found for the selected detection format in the annotations directory.',
              'No file was found')
        return ret, False

    # If detection requires class_id, replace the detection names (integers) by a class from the txt file
    ret = general_utils.replace_id_with_classes(ret, f"{new_folder_gt}/{dataset_name}.names")
    return ret, True


def get_json_df(json_file):
    ds = []
    for file in os.listdir(json_file):
        file_name = os.path.splitext(file)[0]
        f = open(json_file + "/" + file)
        data = json.load(f)
        user_data = data["user"]
        user_data["image"] = file_name
        # print(user_data)
        ds.append(user_data)

    df = pd.DataFrame(ds)
    df.columns = map(str.lower, df.columns)
    return df


def ground_truth_pred_main(new_folder_gt, dataset_name, iteration_count, gt_report_config):
    iteration_count = "Iteration-" + str(iteration_count)
    iou_threshold = 0.50
    load_basic_operation(new_folder_gt)
    det_annotations, passed = load_annotations_det(new_folder_gt, dataset_name)
    if passed is False:
        sys.exit(1)
    # Verify if there are detections
    if det_annotations is None or len(det_annotations) == 0:
        print(
            'No detection of the selected type was found in the folder.\nCheck if the selected type corresponds '
            'to the files in the folder and try again.',
            'Invalid detections')

    gt_annotations = load_annotations_gt(new_folder_gt, dataset_name)
    if gt_annotations is None or len(gt_annotations) == 0:
        print(
            'No ground-truth bounding box of the selected type was found in the folder.\nCheck if the selected '
            'type corresponds to the files in the folder and try again.',
            'Invalid ground truths')
        sys.exit(1)

    # coco_res = get_coco_summary(gt_annotations, det_annotations)

    # mAP = pascal_res['mAP']

    data_df = get_json_df(f"{new_folder_gt}/original Json")
    col_data_val = gt_report_config
    input_val = list(col_data_val.keys())
    input_val.append('image')
    input_val = [x.lower() for x in input_val]
    data_df = data_df.loc[:, data_df.columns.intersection(
        input_val)]
    pascal_res = get_pascalvoc_metrics(gt_annotations, det_annotations, data_df, iou_threshold=iou_threshold,
                                       generate_table=True)

    # print(f"pascal res per class: {pascal_res['per_class']}")

    file_name = f"{new_folder_gt}/reports/class_wise_report.xlsx"

    ################# Requieres Pandas 1.3 and higher only #######################################
    if os.path.isfile(file_name):
        file_size = os.path.getsize(file_name)
        if int(file_size) > 0:
            wb = pd.ExcelFile(file_name)
            writer = pd.ExcelWriter(file_name, engine='openpyxl', mode='a', if_sheet_exists="replace")
        else:
            wb = None
            writer = pd.ExcelWriter(file_name, engine='openpyxl')
    else:
        wb = None
        writer = pd.ExcelWriter(file_name, engine='openpyxl')

    class_res_dict = {"class_name": [], "ap_" + str(iteration_count): [], "precision_" + str(iteration_count): [],
                      "recall_" + str(iteration_count): [], "F1-score_" + str(iteration_count): []}
    all_class_dict = {}
    class_count = 0
    for clsres in pascal_res:
        for c, cat in clsres.items():
            class_count += 1
            classwise_data = {"categories": [], "num_images_" + str(iteration_count): [],
                              "ap_" + str(iteration_count): [],
                              "precision_" + str(iteration_count): [], "recall_" + str(iteration_count): [],
                              "f1-score_" + str(iteration_count): []}

            col_data = gt_report_config
            final_list = list(col_data.values())
            print("final list ", final_list)
            cat_dict = {}

            for lst in list(itertools.product(*final_list)):
                cat_dict['_'.join(lst)] = {}
            for key, val in cat.items():
                cat_dict[key] = val

            num_cat_types = 0

            for k, v in cat_dict.items():
                if bool(v):
                    print(v)
                    num_images = len(v["precision"])
                    prec = v["precision"].mean() if num_images != 0 else 0.0
                    rec = v["recall"].max() if num_images != 0 else 0.0
                    ap = v["AP"]
                    try:
                        f1_score = (2 * prec * rec) / (prec + rec)
                    except ZeroDivisionError:
                        f1_score = 0.0

                    num_cat_types = num_cat_types + 1 if num_images != 0 else num_cat_types

                else:
                    num_images = 0
                    prec = 0.0
                    rec = 0.0
                    ap = 0.0
                    f1_score = 0.0

                classwise_data["categories"].append(k)
                classwise_data["num_images_" + str(iteration_count)].append(num_images)
                classwise_data["ap_" + str(iteration_count)].append(ap)
                classwise_data["precision_" + str(iteration_count)].append(prec)
                classwise_data["recall_" + str(iteration_count)].append(rec)
                classwise_data["f1-score_" + str(iteration_count)].append(f1_score)
        classwise_data["num_cat_types"] = num_cat_types
        all_class_dict[c] = classwise_data

        classwise_df = pd.DataFrame(classwise_data)
        classwise_df.drop('num_cat_types', inplace=True, axis=1)
        class_res_dict["class_name"].append(c)
        class_res_dict["ap_" + str(iteration_count)].append(
            classwise_df["ap_" + str(iteration_count)].sum() / num_cat_types)
        class_res_dict["precision_" + str(iteration_count)].append(
            classwise_df["precision_" + str(iteration_count)].sum() / num_cat_types)
        class_res_dict["recall_" + str(iteration_count)].append(
            classwise_df["recall_" + str(iteration_count)].sum() / num_cat_types)
        class_res_dict["F1-score_" + str(iteration_count)].append(
            classwise_df["f1-score_" + str(iteration_count)].sum() / num_cat_types)
        # import pdb;
        # pdb.set_trace()
        if wb is not None:
            if c in wb.sheet_names:
                print(f"{c} sheet is in excel sheet")
                dataframe2 = pd.read_excel(file_name, sheet_name=c)
                dataframe2.drop("Unnamed: 0", inplace=True, axis=1)
                dataframe2.drop('categories', inplace=True, axis=1)
                print(dataframe2)
                final_df = pd.concat([classwise_df, dataframe2], axis=1, sort=False)
                final_df.to_excel(writer, sheet_name=c)
                print(f'Adding data to existing MASTER excel sheet for {c}')
            else:
                classwise_df.to_excel(writer, sheet_name=c)
                print(f'Adding new sheet to MASTER excel sheet for {c}')
        else:
            classwise_df.to_excel(writer, sheet_name=c)
            print(f'Adding new sheet to MASTER excel sheet for {c}')

    mAP = sum(class_res_dict["ap_" + str(iteration_count)]) / class_count
    class_res_df = pd.DataFrame(class_res_dict)
    if wb is not None:
        if "all Class" in wb.sheet_names:
            dataframe3 = pd.read_excel(file_name, sheet_name="all Class")
            dataframe3.drop("Unnamed: 0", inplace=True, axis=1)
            dataframe3.drop('class_name', inplace=True, axis=1)
            print(dataframe3)
            final_all_df = pd.concat([class_res_df, dataframe3], axis=1, sort=False)
            final_all_df.to_excel(writer, sheet_name="all Class")
            print(f'Adding data to existing MASTER excel sheet for all Class')
        else:
            class_res_df.to_excel(writer, sheet_name="all Class")
            print(f'Adding new sheet to MASTER excel sheet for all Class')
    else:
        class_res_df.to_excel(writer, sheet_name="all Class")
        print(f'Adding new sheet to MASTER excel sheet for all Class')

    writer.close()
    # Coping the generated excel files to respective folders
    shutil.copy2(f"{new_folder_gt}/reports/class_wise_report.xlsx", "gt_reports")

    # # Plotting
    plot_precision_recall_curve(all_class_dict,
                                mAP=mAP, iteration_id=iteration_count,
                                savePath=f"{new_folder_gt}/output",
                                showGraphic=False)
    # Save plots for each class
    plot_precision_recall_curves(all_class_dict,
                                 showAP=True, iteration_id=iteration_count,
                                 savePath=f"{new_folder_gt}/output",
                                 showGraphic=False)
