'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import json
import ast
import pandas as pd
from module import generate_overall_summary as summary_report_generator
from module import generate_detail_report as detail_report_generator
import os
import datetime
from module import confusion_matrix_generator
from module.plot_embeddings import get_template_data, plot_embeddings_in_3d


from sklearn.metrics import precision_score, recall_score, f1_score


def get_prediction_data(prediction_list):
    all_predictions = []
    max_similarity_score  = float('-inf')
    best_match = {}
    next_best_match = {}
    for prediction in prediction_list:
        prediction_data ={}
        bbox = prediction['templateAnnotation']
        prediction_data['class_name'] = prediction['templateClass']
        prediction_data['file_path'] = prediction['templateFile']
        prediction_data['similarity_score'] = float(prediction['similarity_score'])
        prediction_data["bounding_box"]= [ bbox[0], bbox[1], bbox[2], bbox[3]]
        prediction_data["Prediction_result"] = prediction['prediction_result']
        if  prediction_data['similarity_score'] > max_similarity_score:
            best_match = prediction_data
            max_similarity_score = prediction_data['similarity_score']
        all_predictions.append(prediction_data)
    next_best_match = [ap for ap in all_predictions if ap['file_path'] != best_match['file_path']]

    #return best_match,next_best_match,prediction['prediction_result']
    return best_match,next_best_match,best_match.get('Prediction_result', '') 

def get_outcome(gt_class,best_match_class):
    if gt_class == best_match_class:
        return 'correct detection'
    else:
        return 'misprediction'

         
def format_report_data(validation_data):
    report_data = []
    for item in validation_data:
        bbox = item['annotationValues']
        item['GTCategory'] = item['GTCategory'].replace(r' \\','/')
        gt_category = item['GTCategory'][1:].lower().split('/')
        predictions = item['predictions']
        category = {}
        for category_split in gt_category:
            cat= category_split.split('-')
            category[cat[0]] = cat[1]
        predictions=get_prediction_data(item['predictions']) 
        best_match = predictions[0]
        next_best_match = predictions[1]
        prediction_result = predictions[2]
        # Create the concatenated category string dynamically
        concatenated_category = '_'.join([value for key, value in category.items()])


        # Add individual components to the dictionary
        category_dict = {key: value for key, value in category.items()}
        category_dict['category'] = concatenated_category
        report_data.append(
        {
            "image_name": item['fileName'],
            "file_path" : item['filePath'],
            "category": concatenated_category,
            **category_dict,  # Dynamically add all keys from category_dict
            "result" : get_outcome(item['actualClass'],best_match['class_name']),
            "class_name": item['actualClass'],
            "GT":
            {
                "class_name": item['actualClass'],
                "bounding_box": [
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3]
                ],
            } ,
            "predictions":{
                "best_match": best_match,
                "next_best_match": next_best_match,
                "prediction_result" : prediction_result
            }         
        }
        )
    return report_data

def calculate_metrics(data):
    tp = len([d for d in data if d['predictions']["prediction_result"] == "TP"])
    fp = len([d for d in data if d['predictions']["prediction_result"] == "FP"])
    fn = len([d for d in data if d['predictions']["prediction_result"] == "FN"])
    tn = len([d for d in data if d['predictions']["prediction_result"] == "TN"])    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp+tn) / (tp+fp+fn+tn) if (tp+fp+fn+tn) > 0 else 0.0
    return {'data size':len(data),'true positive':tp,'false positive':fp, 'false negative':fn, 'true negative':tn, 'precision': precision, 'recall': recall, 'f1': f1, 'accuracy':accuracy}

def get_data_by_key(data_list,key):
    result = {}
    for data in data_list:
        if data[key] not in result:
            result[data[key]]=[]
        result[data[key]].append(data)
    return result

def get_data_by_two_level_key(data_list,key1,key2):
    result = {}
    for data in data_list:
        if data[key1] not in result:
            result[data[key1]] ={}
        if data[key2] not in result[data[key1]]:
            result[data[key1]][data[key2]] = []
        result[data[key1]][data[key2]].append(data)
    return result



def get_metrics_by_key(data,key):
    key_wise_data = get_data_by_key(data,key)
    result = {}
    for field,field_data in key_wise_data.items():
        metric = calculate_metrics(field_data)
        if field not in result:
            result[field] = []
        result[field]=metric
    return result

def get_metrics_by_two_level_key(data,key1,key2):
    key_wise_data = get_data_by_two_level_key(data,key1,key2)
    result = {}
    for field1,field_l1_data in key_wise_data.items():
        for field2,field1_data in field_l1_data.items():
            metric = calculate_metrics(field1_data)
            if field1 not in result:
                result[field1] = {}
            result[field1][field2]=metric
    return result

def create_timestamp_folder(output_path):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Format and remove last 3 digits of microseconds
    output_folder = os.path.join(output_path,timestamp)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def generate_excel_report(validation_data,report_path, template_folder, prediction_embeddings,row_limit, header_list):
    
    overall_metrics = calculate_metrics(validation_data)
    class_wise_metrics = get_metrics_by_key(validation_data,'class_name')
    category_wise_metrics = get_metrics_by_key(validation_data,'category')
    class_category_wise_metrics=get_metrics_by_two_level_key(validation_data,'class_name','category')


    overall_summary_data = {
        "overall_metrics": overall_metrics,
        "class_wise_metrics" : class_wise_metrics,
        "category_wise_metrics" : category_wise_metrics,
        "class_category_wise_metrics": class_category_wise_metrics
    }

    

    summary_report_generator.export_to_summary_excel(overall_summary_data,report_path, header_list)
    template_data = get_template_data(template_folder)
    result_wise_detail = get_data_by_key(validation_data,'result')
    # for i in range(len(result_wise_detail['misprediction'])):
    for i in range(len(result_wise_detail.get('misprediction', []))):
        item = result_wise_detail['misprediction'][i]
        file_name = item['image_name']
        #interactive_plot_path =plot_embeddings_in_3d(report_path,file_name,prediction_embeddings[file_name],template_data)
        #result_wise_detail['misprediction'][i]['plot_path'] = interactive_plot_path
    
    for result,detail_data in result_wise_detail.items():
        excel_file =os.path.join(report_path,f"{result}.xlsx")
        detail_report_generator.write_data_to_excel(detail_data, excel_file,row_limit)
        print(f"Data written to {excel_file}")

def create_confusion_matrix(validation_data,file_path):
    confusion_matrix = confusion_matrix_generator.create_confusion_matrix(validation_data)
    confusion_matrix_generator.plot_confusion_matrix(confusion_matrix,file_path)


def generate_report(validation_data,val_config, prediction_embeddings):
    report_path = create_timestamp_folder(val_config["output_to"]["path"])
    validation_data = format_report_data(validation_data)
    create_confusion_matrix(validation_data,os.path.join(report_path,"overall_cm.png"))
    template_folder = val_config['template_embeddings']['templatePath']
    row_limit = val_config['output_to']['rows_per_report']
    header_list = val_config['report_generator']['gt_category_headers']
    generate_excel_report(validation_data,report_path, template_folder,prediction_embeddings,row_limit, header_list)
    return report_path

