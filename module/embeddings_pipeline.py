'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import importlib
import io
import os
import traceback
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from module import report_generator
from numba import jit,cuda
from PIL import Image,ImageDraw
from utility.similarity_checks import similarity_check
import utility.constants as const
import xlsxwriter
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
from utility.image_processor import crop_object_from_image, yolo_to_xywh
from module.generate_summary_representation import generate_summary_graph
from module import plot_embeddings


class ValidationByEmbeddingsPipeline:
    def __init__(self, logger, validation_config_dict, report_config_dict):
        self.logger = logger
        self.val_config = validation_config_dict
        self.report_config = report_config_dict
        self.embeddings_model = self.val_config['template_embeddings']['embeddings_model_class']
        self.feature_agg = self.val_config['template_embeddings']['feature_aggregation']
        self.classFilePath = self.val_config['template_embeddings']['classFile']
        self.crop = self.val_config['template_embeddings']['cropObjectFromImages']
        self.input_from_config = self.val_config['input_from']
        self.output_to_config = self.val_config['output_to']
        self.output_to_path = self.val_config['output_to']['path']
        self.rows_per_report = self.val_config['output_to']['rows_per_report']
        self.similarity_check_method = self.val_config['similarity_check']['method']
        self.similarity_records = self.val_config['similarity_check']['records_to_return']
        self.confidence_threshold = self.val_config['similarity_check']['confidence_threshold']        
        self.extension_type = tuple(self.val_config['template_embeddings']['extension_type'])
        
        self.input_from_class = getattr(importlib.import_module("module.data_source"),
                                      self.val_config['input_from']['type'])
        
        self.output_to_class = getattr(importlib.import_module("module.data_source"),
                                      self.val_config['output_to']['type'])

        self.embeddings_model_class = getattr(importlib.import_module("module.embeddings_models"),
                                          self.embeddings_model)
        self.templates = None
        self.classes = None
        self.valDatasetMetadata = None
        self.valDatasetMetadataWithSimilarity = None
    
    def load_template_embeddings(self):
        self.templates= self.embeddings_model_class.load_template_embeddings(path=self.val_config['template_embeddings']['templatePath'])
        return self.templates
    
    def extract_validation_dataset_features(self):
        try:
            #get files
            self.logger.info("Extracting validation dataset images from "+ self.input_from_config['path'])         
            folderInfoList = self.input_from_class.getFilesWithFolderName(self,self.input_from_config['path'],self.extension_type)

            self.logger.info("Validation dataset contains " + str(len(folderInfoList)) + " categories")     

            #get classes names
            # absClassesPath = os.path.join( os.getcwd() + self.input_from_config['path']+"/classes.names")
            # classesFile = open(absClassesPath,"r")
            absClassesPath = self.input_from_config['path']+"/classes.names"
            classesFile = open(absClassesPath,"r")
            classesFileContent = classesFile.read()
            classes = classesFileContent.splitlines()
            self.classes = classes
            self.logger.info("Validation dataset contains " + str(len(classes)) + " classes")     

            
            #extract features
            valDatasetMetadata =[]
            encoder = self.embeddings_model_class()
            for folder in folderInfoList:   
                featureList =[]
                for fileInfo in folder['files_list']:      
                    self.logger.info(f"file {fileInfo}")
                    fileName = os.path.basename(fileInfo)
                    img =  Image.open(fileInfo).convert("RGB")
                    
                    #get actual class name
                    annoationFile = open(os.path.splitext(fileInfo)[0]+'.txt',"r")
                    annotationInfo = annoationFile.read()
                    annoationFile.close()
                    actualClassId = int(annotationInfo.split(' ')[0])
                    actualClass=classes[actualClassId]

                    # Parse the annotation information
                    annotationValues = list(map(float, annotationInfo.split()))
                    class_id, *values = annotationValues

                    if(self.crop):                    
                        imgCropinfo = yolo_to_xywh(annotationInfo,img.width,img.height)
                        annotationInfo = imgCropinfo
                        croppedImage = crop_object_from_image(fileInfo,img,imgCropinfo,None,None,False, False)
                        featureEmbedding = encoder.extract_features(image=croppedImage,aggregation=self.val_config['template_embeddings']['feature_aggregation'])
                    else:
                        annotationInfo= values
                        featureEmbedding = encoder.extract_features(image=img,aggregation=self.val_config['template_embeddings']['feature_aggregation'])
                    img.close()
                    featureList.append({"fileName":fileName,"filePath":fileInfo,"annotationValues":annotationInfo,"embeddings":featureEmbedding,"actualClass":actualClass,"actualClassId":actualClassId})   
                if(len(featureList) > 0):
                    self.logger.info("Validation dataset : GT Category"+folder['folderName']+" number of features: "+ str(len(featureList)))     
                    valDatasetMetadata.append({"GTCategory":folder['folderName'], "filesList":featureList})  

            self.logger.info("Feature extraction of validation dataset images completed")       
            self.valDatasetMetadata = valDatasetMetadata
        except Exception as e:
            self.logger.error(e)
            traceback.print_exc()    

    def check_similarity_score(self):
        valDatasetMetadataWithSimilarity = []
        for category in self.valDatasetMetadata:   
            featureList =[]
            for validationFile in category['filesList']: 
                #self.logger.info("Feature comparision for " + feature['fileName'])       
                predictions =[]
                scores=[] 
                templateFiles = []
                for templateFile in self.templates:                    
                    #self.logger.info(template['name'] +" with " + str(len(template['embeddings'])) + " embeddings" )   
                    for template in templateFile['embeddings']: 
                        score = similarity_check(source=template['embeddings'], target=validationFile['embeddings'], method=self.similarity_check_method)
                        scores.append(score)
                        prediction_result = {
                            "templateFile": template['filePath'],
                            "templateAnnotation": template['bBox'],
                            "templateClass": template['originalClass'],
                            "templateClassId": template['originalClassId'],
                            "similarity_score": score,
                            "confidence_score": score, 
                            "prediction_result": None 
                        }
                        predictions.append(prediction_result)
                
                if(self.feature_agg == const.MAX):
                    maxScoreIndex = np.argmax(scores)
                
                # Sort by similarity score
                predictions = sorted(predictions, key=lambda k: k['similarity_score'], reverse=True)
                
                # Get top N predictions based on config
                finalPredictions = predictions[:self.similarity_records]
                
                # Evaluate prediction results based on confidence threshold
                for pred in finalPredictions:
                    is_same_class = int(pred['templateClassId']) == int(validationFile['actualClassId'])
                    exceeds_threshold = float(pred['similarity_score']) >= float(self.confidence_threshold)
                    if is_same_class and exceeds_threshold:
                        pred['prediction_result'] = "TP"
                    elif not is_same_class and exceeds_threshold:
                        pred['prediction_result'] = "FP"
                    elif is_same_class and not exceeds_threshold:
                        pred['prediction_result'] = "FN"
                    elif not is_same_class and not exceeds_threshold: # not same_class and not exceeds_threshold
                        pred['prediction_result'] = "TN"
                    
                
                validation_result = {
                    "GTCategory": category['GTCategory'],
                    "filePath": validationFile['filePath'],
                    "fileName": validationFile['fileName'],
                    "annotationValues": validationFile['annotationValues'],
                    "actualClass": validationFile['actualClass'],
                    "actualClassId": validationFile['actualClassId'],
                    "predictions": finalPredictions
                }
               
                valDatasetMetadataWithSimilarity.append(validation_result)
                self.logger.info(f"Similarity check completed with confidence threshold: {self.confidence_threshold}")
        self.valDatasetMetadataWithSimilarity = valDatasetMetadataWithSimilarity

    def generate_validation_report (self):

        prediction_embeddings = {}
        for category_data in self.valDatasetMetadata:
            for embedding_data in category_data['filesList']:
                prediction_embeddings[embedding_data['fileName'] ]=embedding_data['embeddings']  
        output_report_path = report_generator.generate_report(self.valDatasetMetadataWithSimilarity,self.val_config,prediction_embeddings)
        generate_summary_graph(output_report_path, self.val_config)



        
class FeatureExtractionPipeline:
    def __init__(self, logger, validation_config_dict, report_config_dict):
        self.logger = logger
        self.val_config = validation_config_dict
        self.report_config = report_config_dict
        self.embeddings_model = self.val_config['template_embeddings']['embeddings_model_class']
        self.feature_agg = self.val_config['template_embeddings']['feature_aggregation']
        self.crop = self.val_config['template_embeddings']['cropObjectFromImages']
        self.input_from_config = self.val_config['input_from']
        self.output_to_config = self.val_config['output_to']
        self.classFilePath = self.val_config['input_from']['classFile']
        self.output_to_path = self.val_config['output_to']['path']
        self.similarity_check_method = self.val_config['similarity_check']['method']
        self.extension_type = tuple(self.val_config['template_embeddings']['extension_type'])
        self.saveCroppedObject = self.val_config['template_embeddings']['saveCroppedObject']
        self.saveCroppedObjectWithBbox = self.val_config['template_embeddings']['saveCroppedObjectWithBbox']

        self.input_from_class = getattr(importlib.import_module("module.data_source"),
                                      self.val_config['input_from']['type'])
        
        self.output_to_class = getattr(importlib.import_module("module.data_source"),
                                      self.val_config['output_to']['type'])

        self.embeddings_model_class = getattr(importlib.import_module("module.embeddings_models"),
                                          self.embeddings_model)
        self.templates = None
        self.valDatasetMetadata = None
        self.valDatasetMetadataWithSimilarity = None
    
    
    def extract_and_save_features(self):
        # Import the plotting functions from plot.py
        #from plot import get_template_data, plot_embeddings_in_3d, plot_pca_from_dict_list, plot_plotly_3d_pca
        
        #segregate the images using knn    
        encoder = self.embeddings_model_class()    
        noOfImages = 20
        #profileTemplateImages = encoder.get_template_images_by_knn(folder_path = self.input_from_config['path'],num_images=noOfImages)

        folderInfoList = self.input_from_class.getFilesWithFolderName(self,self.input_from_config['path'],self.extension_type)
        #get classes names
        # absClassesPath = os.path.join( os.getcwd() + self.classFilePath)
        # classesFile = open(absClassesPath,"r")
        classesFile = open(self.classFilePath,"r")
        classesFileContent = classesFile.read()
        classes = classesFileContent.splitlines()
        self.logger.info("Template dataset contains " + str(len(classes)) + " classes")   

        all_features = []
        
        for folder in folderInfoList:  
            featureList = []
            for file in folder['files_list']:     
                #get actual class name
                annoationFile = open(os.path.splitext(file)[0]+'.txt',"r")
                annotationInfo = annoationFile.read()
                annoationFile.close()
                actualClassId = int(annotationInfo.split(' ')[0])
                actualClass=classes[actualClassId] 
                # Parse the annotation information
                annotationValues = list(map(float, annotationInfo.split()))
                class_id, *values = annotationValues
                # Check if the annotation is in YOLO format
                is_yolo = True
                for value in values:
                    if value < 0 or value > 1:  # Normalized values must be between 0 and 1
                        is_yolo = False
                        break
                # Load the image
                img = Image.open(file).convert("RGB")
                imgCropInfo = None
                if self.crop:
                    if is_yolo:
                        # Convert YOLO format to XYWH
                        imgCropInfo = yolo_to_xywh(annotationInfo,img.width,img.height)
                    else:
                        # Use the values as they are for XYWH format
                        print(file)
                        x_min, y_min, width, height = values
                        imgCropInfo = (x_min, y_min, width, height)
                    # Crop the image using the calculated or existing crop info
                    img = crop_object_from_image(file,img, imgCropInfo,self.output_to_path,os.path.basename(folder['folderName']),self.saveCroppedObject,self.saveCroppedObjectWithBbox)
                    
                    
                #Extract features of the given image/cropped image
                features = encoder.extract_features(
                    image=img,
                    aggregation=self.val_config['template_embeddings']['feature_aggregation']
                )
                img.close()
                featureList.append({"parentFolder": folder['parentFolder'], "folderName" : folder['folderName'],"originalClass":actualClass,"originalClassId":actualClassId,"filePath": file, "embeddings" : features,"bBox":imgCropInfo})
                #featureList.append({"parentFolder": folder['parentFolder'], "Foldername" : folder['folderName'],"files_list": file, "features" : features})
                
                # Generate 3D plot for this specific feature
                
                
            if(len(featureList) > 0):
                fileName = str(folder['folderName']).replace("\\","")
                self.logger.info(folder['parentFolder'] + "pickle file is created with " + str(len(featureList)) + " features" )  
                encoder.save_template_embeddings(featureList,fileName, self.output_to_path,'pickle')
                file_name = os.path.basename(file)
                template_data = plot_embeddings.get_template_data(self.output_to_path)
                #print("template data:", template_data)
                plot_embeddings.plot_embeddings_in_3d(self.output_to_path, file_name, features, template_data)
                self.logger.info(f"Created 3D PCA plot for {file_name}")
                all_features.extend(featureList)
                
        # Also create a combined plot for all features
        if all_features:
            template_data = plot_embeddings.get_template_data(self.output_to_path)
            pca_plot_data = plot_embeddings.plot_pca_from_dict_list(all_features, n_components=3)
            pca = pca_plot_data[0]
            classes = pca_plot_data[1]
            combined_plot_path = os.path.join(self.output_to_path, "3d_plot", "all_classes_combined.html")
            plot_embeddings.plot_plotly_3d_pca(pca, classes, combined_plot_path)
            self.logger.info(f"Created combined 3D PCA plot: {combined_plot_path}")
                
        return featureList