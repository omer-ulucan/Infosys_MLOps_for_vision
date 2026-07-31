'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import datetime
import glob
import inspect
import json
import os
import random
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid
import warnings
import zipfile
from decimal import *
from distutils.dir_util import copy_tree
from os.path import exists
from pathlib import PurePath
from shutil import copy

import albumentations as A
import cv2
import mlflow
import requests
import xlsxwriter
import psutil
#from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
#from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
#from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, \
#    Region
from minio import Minio
from msrest.authentication import ApiKeyCredentials
from pylabel import importer
import torch
import torchvision.ops.boxes as bops
from utility.json_ops import get_dict_from_json
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


# class AzureCustomVision:
#     """ AZURE CUSTOM VISION TRAINING PIPELINE BASED ON PYTHON SDK MAIN FILE """

#     def __init__(self, azure_config_dict, custom_logger, gt_report_config_dict, custom_pipeline_yolo_version=None):
#         self.custom_logger = custom_logger
#         self.azure_config_dict = azure_config_dict
#         self.gt_report_config_dict = gt_report_config_dict
#         self.custom_pipeline_yolo_version = custom_pipeline_yolo_version
#         self.training_endpoint = self.azure_config_dict.get("training_endpoint")
#         self.training_key = self.azure_config_dict.get("training_key")
#         self.pred_endpoint = self.azure_config_dict.get("prediction_endpoint")
#         self.prediction_key = self.azure_config_dict.get("prediction_key")
#         self.prediction_resource_id = self.azure_config_dict.get("prediction_resource_id")
#         self.publish_iteration_name = self.azure_config_dict.get("publish_iteration_name")
#         self.project_name = self.azure_config_dict.get("project_name")
#         self.domain_type = self.azure_config_dict.get("domain_type")
#         self.training_type = self.azure_config_dict.get("training_type")
#         self.reserved_budget_in_hours = self.azure_config_dict.get("reserved_budget_in_hours")
#         self.domain_name = self.azure_config_dict.get("domain_name")
#         self.dataset_name = self.azure_config_dict.get("labels_file_name")
#         self.tag_list = [line.strip() for line in open(f"training_data/labels/{self.dataset_name}.names", 'r')]
#         self.probability_threshold_data = self.azure_config_dict.get("probability_threshold")
#         self.overlap_threshold_data = self.azure_config_dict.get("overlap_threshold")
#         self.label_path = self.azure_config_dict.get("label_path")
#         self.img_path = self.azure_config_dict.get("img_path")
#         self.platform_val = self.azure_config_dict.get("platform")
#         self.metadata_path = self.azure_config_dict.get("metadata_path")
#         self.test_image_location = self.azure_config_dict.get("test_image_location")
#         self.augmentation_of_data = self.azure_config_dict.get("augmentation_of_data")
#         self.augmentation_of_data_type = self.azure_config_dict.get("augmentation_of_data_type")
#         self.special_augmentation_of_data = self.azure_config_dict.get("special_augmentation_of_data")
#         self.notification_email = self.azure_config_dict.get("notification_email")
#         self.endpoint_status = self.azure_config_dict.get("endpoint_status")
#         self.batch_upload = self.azure_config_dict.get("upload_type")

#         self.credentials = None
#         self.trainer = None
#         self.prediction_credentials = None
#         self.predictor = None
#         self.project_data = None
#         self.project = None
#         self.tags = None
#         self.tag_value = None
#         self.dataset = None
#         self.overall_performance = None
#         self.training_data_types_part1 = None
#         self.training_data_types_part2 = None
#         self.transforms = None
#         self.special_transforms = None
#         self.iteration = None
#         self.export = None
#         self.exports = None
#         self.prediction_dict = None
#         self.prediction_list_final = None
#         self.publish_iteration_name_final = None
#         self.msg = None
#         self.obj_detection_domain = None
#         self.project_dl = None
#         self.img_path_data = None
#         self.performance = None
#         self.tag_performance_data = None
#         self.uuid_custom = None
#         self.ground_truth_subfolders = None
#         self.class_dataset = None
#         self.total_images = None
#         self.new_folder_gt = None
#         self.length_augmentation_list = None
#         self.labels_file_path = None
#         self.iteration_count = 0

#     def perform_augmentation(self):

#         """ Perform Special Augmentation on user provided classes and generic Augmentation on all classes"""
#         print("Current Iteration has following classes : ", self.tag_list)
#         print()
#         special_transforms_class = input("Please Enter Classes from above list only on which you want to perform "
#                                          "special Augmentation on with ',' "
#                                          "separated values  (Input is Case sensitive) or NA if you want to perform "
#                                          "None : ")
#         try:
#             special_class_index_list = []
#             if special_transforms_class:
#                 special_transforms_class_list = list(special_transforms_class.split(","))
#                 # Getting index for all labels provided by user
#                 for special_class in special_transforms_class_list:
#                     if special_class in self.tag_list:
#                         special_class_index_list.append(self.tag_list.index(special_class))
#                     else:
#                         pass
#                 special_class_index_string = map(str, special_class_index_list)
#                 special_class_index_list = list(special_class_index_string)
#                 print("Index for which special Augmentation to be performed", special_class_index_list)
#             else:
#                 special_class_index_list = []

#         except Exception as e:
#             special_class_index_list = []

#         print(special_class_index_list)

#         def write_image_box(image_write, bboxes_write, filename_write):
#             """ Generate Augmented images and text files"""
#             cv2.imwrite("training_data/Images/items/" + filename_write + ".jpg", image_write[:, :, ::-1])
#             with open("training_data/labels/yolo/" + filename_write + ".txt", 'w+') as file:
#                 for box in bboxes_write:
#                     x, y, w, h, category = box
#                     file.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
#             file.close()

#         # IMPORT ONLY IN YOLO FORMAT

#         # SPECIAL AUGMENTATION on User specific classes
#         try:
#             # Special data Augmentations performed or not on user specified class
#             # See a list of transforms performed by Albumentations.
#             # https://albumentations.ai/docs/getting_started/transforms_and_targets/
#             if self.special_augmentation_of_data and special_class_index_list:
#                 print('Performing Special Augmentations on classes specified by the User')
#                 self.custom_logger.info('Performing Special Augmentations on classes specified by the User')
#                 self.special_transforms = []
#                 for special_aug_item in self.special_augmentation_of_data:
#                     for special_aug_key, special_aug_val in special_aug_item.items():
#                         special_aug_val["always_apply"] = True
#                         special_aug_data = [getattr(A, special_aug_key)(**special_aug_val)]
#                         self.special_transforms.append(special_aug_data)
#                 print("Special Aug list", self.special_transforms)
#                 self.custom_logger.info(f"Special Aug list {self.special_transforms}")

#                 self.labels_file_path = glob.glob(r'training_data/labels/yolo/*.txt')

#                 special_aug_files = []

#                 for txt_file in self.labels_file_path:
#                     with open(txt_file) as fp:
#                         first_index = zip(*[line.rstrip().split(' ') for line in fp])
#                         first_list = list(first_index)
#                         first_element = first_list[0]
#                         valid_files = any(i in first_element for i in special_class_index_list)
#                         if valid_files:
#                             special_aug_files.append(txt_file)
#                         else:
#                             pass

#                 def do_split(file_path_data):
#                     """ Perform split for filenames """
#                     file_val = os.path.basename(file_path_data)
#                     return os.path.splitext(file_val)[0]

#                 # print(special_aug_files)
#                 special_aug_images = list(map(do_split, special_aug_files))
#                 # print(special_aug_images)

#                 for special_images in special_aug_images:
#                     try:
#                         all_files = os.listdir("training_data/Images/items")
#                         r = re.compile(f"{special_images}.(png|jpg)")
#                         mapped_img = list(filter(r.match, all_files))
#                         if mapped_img:
#                             working_img = mapped_img[0]
#                             special_filename = os.path.splitext(working_img)[0]
#                             special_image = cv2.imread("training_data/Images/items/" + working_img)
#                             special_image = cv2.cvtColor(special_image, cv2.COLOR_BGR2RGB)
#                             annotation_path = "training_data/labels/yolo/" + str(special_filename) + ".txt"
#                             lines = [line.rstrip('\n') for line in open(annotation_path)]
#                             bboxes = []
#                             if lines != ['']:
#                                 for line in lines:
#                                     category, x, y, w, h = line.split(" ")
#                                     bboxes.append([float(x), float(y), float(w), float(h), int(category)])

#                             for special_index, special_aug in enumerate(self.special_transforms):
#                                 special_transforms = A.Compose(special_aug,
#                                                                bbox_params=A.BboxParams(format="yolo", min_area=1,
#                                                                                         min_visibility=0.5))
#                                 special_res = special_transforms(image=special_image, bboxes=bboxes)
#                                 special_filename_res = special_filename + "_special_" + str(special_index)
#                                 write_image_box(special_res["image"], special_res["bboxes"], special_filename_res)
#                         else:
#                             pass
#                     except Exception as e:
#                         print(e)
#                         self.custom_logger.exception(e)
#                         continue
#                 print('Successfully Performed Special Augmentations on user specified classes')
#                 self.custom_logger.info('Successfully Performed Special Augmentations on user specified classes')
#             else:
#                 print('NO Special Augmentations performed')
#                 self.custom_logger.info('NO Special Augmentations performed')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         def split_image_data(a, n):
#             k, m = divmod(len(a), n)
#             return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

#         # GENERIC AUGMENTATION on all classes or semi on all classes
#         try:
#             # Standard Generic data Augmentations performed on all class or semi on all classes
#             # See a list of transforms performed by Albumentations.
#             # https://albumentations.ai/docs/getting_started/transforms_and_targets/
#             if self.augmentation_of_data:
#                 self.transforms = []
#                 for aug_item in self.augmentation_of_data:
#                     for aug_key, aug_val in aug_item.items():
#                         aug_val["always_apply"] = True
#                         aug_data = [getattr(A, aug_key)(**aug_val)]
#                         self.transforms.append(aug_data)
#                 print("Generic Aug list", self.transforms)
#                 self.custom_logger.info(f"Generic Aug list{self.transforms}")

#                 if self.augmentation_of_data_type == "Full":
#                     print('Performing Generic Standard Augmentations on all classes  as specified by the User')
#                     self.custom_logger.info('Performing Generic Standard Augmentations on all classes  as specified by '
#                                             'the User')
#                     # Give each aug as list separated as given below 4 augmentation are performed

#                     for file in os.listdir("training_data/Images/items"):
#                         try:
#                             if file.endswith(('.png', '.jpg', '.jpeg')):
#                                 filename = os.path.splitext(file)[0]
#                                 # print(os.path.abspath(file))
#                                 image = cv2.imread("training_data/Images/items/" + file)
#                                 image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                 annotation_path = "training_data/labels/yolo/" + str(filename) + ".txt"
#                                 lines = [line.rstrip('\n') for line in open(annotation_path)]
#                                 bboxes = []
#                                 if lines != ['']:
#                                     for line in lines:
#                                         category, x, y, w, h = line.split(" ")
#                                         bboxes.append([float(x), float(y), float(w), float(h), int(category)])

#                                 for index, aug in enumerate(self.transforms):
#                                     transforms = A.Compose(aug,
#                                                            bbox_params=A.BboxParams(format="yolo", min_area=1,
#                                                                                     min_visibility=0.5))
#                                     res = transforms(image=image, bboxes=bboxes)
#                                     filename_res = filename + "_" + str(index)
#                                     write_image_box(res["image"], res["bboxes"], filename_res)
#                             else:
#                                 pass
#                         except Exception as e:
#                             print(e)
#                             self.custom_logger.exception(e)
#                             continue
#                     print('Successfully Performed Standard Generic Augmentations on all classes as specified by the '
#                           'User')
#                     self.custom_logger.info('Successfully Performed Standard Generic Augmentations on all classes as '
#                                             'specified by the User')
#                 else:
#                     print('Performing Generic Standard Augmentations on parts of classes only to have reduced data '
#                           'volume')
#                     self.custom_logger.info(
#                         'Performing Generic Standard Augmentations on parts of classes only to have reduced data '
#                         'volume')
#                     self.length_augmentation_list = len(self.transforms)
#                     self.labels_file_path = glob.glob(r'training_data/labels/yolo/*.txt')
#                     for list_index in range(len(self.tag_list)):
#                         print(f"Augmentation started for class {self.tag_list[list_index]}")
#                         self.custom_logger.info(f"Augmentation started for class {self.tag_list[list_index]}")
#                         images_list = []
#                         for txt_file in self.labels_file_path:
#                             txt_filename = os.path.basename(txt_file)
#                             with open(txt_file) as fp:
#                                 first_index = zip(*[line.rstrip().split(' ') for line in fp])
#                                 first_list = list(first_index)
#                                 first_element = first_list[0]
#                                 # print(first_element)
#                                 if str(list_index) in first_element:
#                                     images_list.append(txt_filename)
#                                 else:
#                                     pass
#                         print(f"Img list length class vise{len(images_list)}")
#                         self.custom_logger.info(f"Img list length class vise{len(images_list)}")
#                         random.shuffle(images_list)
#                         augmentation_list_class_vise = (list(split_image_data(images_list,
#                                                                               self.length_augmentation_list)))
#                         print(f"augmentation_list_class_vise {augmentation_list_class_vise}")
#                         self.custom_logger.info(f"augmentation_list_class_vise {augmentation_list_class_vise}")
#                         for image_chunk_count, image_chunk_list in enumerate(augmentation_list_class_vise):
#                             print(f"Augmentation performed {self.transforms[image_chunk_count]}")
#                             self.custom_logger.info(f"Augmentation performed {self.transforms[image_chunk_count]}")
#                             for image_file in image_chunk_list:
#                                 image_filename = os.path.splitext(image_file)[0]
#                                 all_files = os.listdir("training_data/Images/items")
#                                 r = re.compile(f"{image_filename}.(png|jpg)")
#                                 mapped_img = list(filter(r.match, all_files))
#                                 if mapped_img:
#                                     try:
#                                         image = cv2.imread(f"training_data/Images/items/{mapped_img[0]}")
#                                         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                                         annotation_path = "training_data/labels/yolo/" + str(image_filename) + ".txt"
#                                         lines = [line.rstrip('\n') for line in open(annotation_path)]
#                                         bboxes = []
#                                         if lines != ['']:
#                                             for line in lines:
#                                                 category, x, y, w, h = line.split(" ")
#                                                 bboxes.append([float(x), float(y), float(w), float(h), int(category)])

#                                         transforms = A.Compose(self.transforms[image_chunk_count],
#                                                                bbox_params=A.BboxParams(format="yolo", min_area=1,
#                                                                                         min_visibility=0.5))
#                                         res = transforms(image=image, bboxes=bboxes)
#                                         filename_res = image_filename + "_" + str(image_chunk_count)
#                                         write_image_box(res["image"], res["bboxes"], filename_res)
#                                     except Exception as e:
#                                         print(e)
#                                         self.custom_logger.exception(e)
#                                         continue
#                                 else:
#                                     print(f"No Image with name exist {mapped_img[0]}")
#                                     self.custom_logger.info(f"No Image with name exist {mapped_img[0]}")
#                         print(f"Augmentation completed for class {self.tag_list[list_index]}")
#                         self.custom_logger.info(f"Augmentation completed for class {self.tag_list[list_index]}")
#                     print('Successfully Performed Generic Standard Augmentations on parts of classes only to have '
#                           'reduced data ')
#                     self.custom_logger.info('Successfully Performed Generic Standard Augmentations on parts of '
#                                             'classes only to have reduced data')
#             else:
#                 print('NO Generic Standard Augmentations performed')
#                 self.custom_logger.info('NO Generic Standard Augmentations performed')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#     def perform_pipeline_operation(self):
#         """ Function to perform all activities performed in custom vision UI through python SDK script which can be
#         scheduled using manual or time scheduler and involves reporting and other tasks """
#         # Provide details for connecting to azure custom vision
#         # Replace with valid values

#         self.credentials = ApiKeyCredentials(in_headers={"Training-key": self.training_key})
#         self.trainer = CustomVisionTrainingClient(self.training_endpoint, self.credentials)
#         self.prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": self.prediction_key})
#         self.predictor = CustomVisionPredictionClient(self.pred_endpoint, self.prediction_credentials)

#         # ## Project name for use case

#         # Create Project if not exists and use same if existing project is used
#         # USING GENERAL DOMAIN NAME IF TO BE PUBLISHED AS API ENDPOINT AND IF EXPORTING REQUIRED USE DOMAIN NAME GENERAL
#         # COMPACT MODE
#         try:
#             self.project_data = next(filter(lambda p: p.name == self.project_name, self.trainer.get_projects()), None)

#             if not self.project_data:
#                 # Find the object detection domain
#                 self.obj_detection_domain = next(domain for domain in self.trainer.get_domains() if
#                                                  domain.type == self.domain_type and domain.name == self.domain_name)

#                 # Create a new project
#                 print("Creating project ...." + " " + self.project_name)
#                 self.custom_logger.info("Creating project ...." + " " + self.project_name)
#                 self.project = self.trainer.create_project(self.project_name, domain_id=self.obj_detection_domain.id)
#                 print("Project Created Successfully ....")
#                 self.custom_logger.info("Project Created Successfully ....")
#             else:
#                 print("Project already exist ...." + " " + self.project_name)
#                 self.custom_logger.info("Project already exist ...." + " " + self.project_name)
#                 self.project = self.project_data
#                 print("Utilizing Existing Project ....")
#                 self.custom_logger.info("Utilizing Existing Project ....")

#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         try:
#             self.tags = dict();
#             # ## Create Tags if not exist and use same if existing Tags are used
#             for tag_data in self.tag_list:
#                 tag_data_val = next(filter(lambda t: t.name == tag_data, self.trainer.get_tags(self.project.id)), None)
#                 if not tag_data_val:
#                     self.tag_value = self.trainer.create_tag(self.project.id, tag_data)
#                     print("Creating New Tags ....")
#                     self.custom_logger.info("Creating New Tags ....")
#                 else:
#                     self.tag_value = tag_data_val
#                     print("Using Existing Tags ....")
#                     self.custom_logger.info("Using Existing Tags ....")
#                 # Creating Dictionary of Tag Name and its location
#                 self.tags[tag_data] = self.tag_value

#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         def split(list_a, chunk_size):
#             try:
#                 for i in range(0, len(list_a), chunk_size):
#                     yield list_a[i:i + chunk_size]
#             except Exception as e:
#                 print(e)
#                 self.custom_logger.exception(e)

#         # Upload to Custom Vision
#         try:
#             # Create PyLabel Dataset
#             print("Creating Dataset for upload ....")
#             self.custom_logger.info("Creating Dataset for upload ....")
#             # Iterate the rows for each image in the dataframe
#             # Import annotations as a PyLabel dataset
#             self.dataset = importer.ImportYoloV5(path=self.label_path,
#                                                  path_to_images=self.img_path,
#                                                  cat_names=self.tag_list
#                                                  )
#             # Checking data set information
#             print("Created Dataset for upload ....")
#             self.custom_logger.info("Created Dataset for upload ....")
#             print("Actual Training data Details")
#             self.custom_logger.info("Actual Training data Details")
#             print("Number of images: ", self.dataset.analyze.num_images)
#             self.custom_logger.info(f"Number of images: {self.dataset.analyze.num_images}")
#             self.total_images = self.dataset.analyze.num_images
#             print("Distribution of classes:", self.dataset.analyze.class_counts)
#             self.custom_logger.info(f"Distribution of classes: {self.dataset.analyze.class_counts}")
#             # Check for last iteration details
#             # Move image to temporary folder
#             isExist = os.path.exists("training_data_complete")
#             if isExist:
#                 print("Images Upload failed in last run continuing from last upload progress")
#                 self.custom_logger.info("Images Upload failed in last run continuing from last upload progress")
#             else:
#                 os.mkdir("training_data_complete")
#                 os.mkdir("training_data_failed")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         # Class vise Batch Upload mode by splitting on classes
#         if self.batch_upload == "Class":
#             # Class vise Batch Upload mode by splitting on classes
#             t1 = time.localtime()
#             current_time_1 = time.strftime("%H:%M:%S", t1)
#             print("Upload Start Time :", current_time_1)
#             self.custom_logger.info(f"Upload Start Time : {current_time_1}")
#             print("Batch Upload mode")
#             self.custom_logger.info("Batch Upload mode")
#             dfs = dict(tuple(self.dataset.df.groupby('cat_name')))
#             dfs_list = list(dfs.keys())
#             print("Upload Started ....")
#             self.custom_logger.info("Upload Started ....")
#             # Clearing if any empty classes are their in batch
#             dfs_list_final = [i for i in dfs_list if i]
#             for tag in dfs_list_final:
#                 print(f'Upload started for {tag}')
#                 self.custom_logger.info(f'Upload started for {tag}')
#                 self.class_dataset = dfs[tag]
#                 print(self.class_dataset.shape)
#                 self.custom_logger.info(f"Shape : {self.class_dataset.shape}")
#                 output_list = []
#                 for img_filename, img_df in self.class_dataset.groupby('img_filename'):
#                     img_path = str(
#                         PurePath(self.dataset.path_to_annotations, str(img_df.iloc[0].img_folder),
#                                  img_filename))
#                     assert exists(img_path), f"File does not exist: {img_path}"
#                     # Create a region object for each bounding box in the dataset
#                     regions = []
#                     for index, row in img_df.iterrows():
#                         # Normalize the bounding box coordinates between 0 and 1
#                         x = Decimal(row.ann_bbox_xmin / row.img_width).min(1)
#                         y = Decimal(row.ann_bbox_ymin / row.img_height).min(1)
#                         w = Decimal(row.ann_bbox_width / row.img_width).min(1 - x)
#                         h = Decimal(row.ann_bbox_height / row.img_height).min(1 - y)

#                         regions.append(Region(
#                             tag_id=self.tags[row.cat_name].id,
#                             left=x,
#                             top=y,
#                             width=w,
#                             height=h
#                         )
#                         )

#                     # Create an objects with the all images and all of the annotations for all images in 1 batch
#                     with open(img_path, mode="rb") as image_contents:
#                         image_and_annotations = ImageFileCreateEntry(name=img_filename,
#                                                                      contents=image_contents.read(),
#                                                                      regions=regions)
#                         output_list.append(image_and_annotations)
#                 upload_list = (list(split(output_list, 60)))

#                 # Upload the images and all annotations for that images with of 60 images
#                 for batch in upload_list:
#                     print(len(batch))
#                     retry_limit = 0
#                     while retry_limit < 5:
#                         try:
#                             upload_status = "Completed"
#                             upload_result = self.trainer.create_images_from_files(
#                                 self.project.id,
#                                 ImageFileCreateBatch(images=batch)
#                             )
#                         except Exception as e:
#                             print(e)
#                             self.custom_logger.exception(e)
#                             print("No. of retry count :", retry_limit)
#                             self.custom_logger.info(f"No. of retry count : {retry_limit}")
#                             print('Sleeping for 120 seconds')
#                             self.custom_logger.info('Sleeping for 120 seconds')
#                             time.sleep(120)
#                             retry_limit += 1
#                             continue
#                         break
#                     else:
#                         upload_status = "Aborted"
#                         self.custom_logger.info(f"Upload {upload_status} ....")
#                         print(f"Upload {upload_status} ....")
#                         sys.exit(1)
#                     time.sleep(0.2)
#                     # If upload is not successful, print details about that image for debugging
#                     image_status_dict = dict();
#                     if not upload_result.is_batch_successful:
#                         print("Image upload failed.")
#                         self.custom_logger.info("Image upload failed.")
#                         for image in upload_result.images:
#                             image_status_dict[image.source_url] = image.status
#                     else:
#                         print("Image upload completed.")
#                         self.custom_logger.info("Image upload completed.")
#                         for image in upload_result.images:
#                             image_status_dict[image.source_url] = image.status
#                     for image_key, status_value in image_status_dict.items():
#                         if status_value == "OK":
#                             try:
#                                 shutil.move(f"training_data/Images/items/{image_key}", "training_data_complete/")
#                                 general_name = os.path.splitext(image_key)[0]
#                                 shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_complete/")
#                             except Exception as e:
#                                 print(e)
#                                 self.custom_logger.exception(e)
#                         else:
#                             try:
#                                 shutil.move(f"training_data/Images/items/{image_key}", "training_data_failed/")
#                                 general_name = os.path.splitext(image_key)[0]
#                                 shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_failed/")
#                                 print(image_key, status_value)
#                                 self.custom_logger.info(f"{image_key}: {status_value}")
#                             except Exception as e:
#                                 print(e)
#                                 self.custom_logger.exception(e)
#                 print(f'Upload completed for {tag}')
#                 self.custom_logger.info(f'Upload completed for {tag}')
#             # This will take a few minutes
#             print(f"Upload {upload_status} ....")
#             self.custom_logger.info(f"Upload {upload_status} ....")
#             t2 = time.localtime()
#             current_time_2 = time.strftime("%H:%M:%S", t2)
#             print("Upload End Time :", current_time_2)
#             self.custom_logger.info(f"Upload End Time :{current_time_2}")

#         # Batch Upload mode with capability for MOC data
#         elif self.batch_upload == "Batch":
#             # Batch Upload mode with capability for MOC data
#             print("Batch Upload mode with capability for MOC data")
#             self.custom_logger.info("Batch Upload mode with capability for MOC data")
#             upload_status = "Completed"
#             t1 = time.localtime()
#             current_time_1 = time.strftime("%H:%M:%S", t1)
#             print("Upload Start Time :", current_time_1)
#             self.custom_logger.info(f"Upload Start Time : {current_time_1}")
#             print("Upload Started ....")
#             self.custom_logger.info("Upload Started ....")
#             batch_list = self.dataset.df.groupby('img_filename')
#             batch_list_keys = list(batch_list.groups.keys())
#             batch_upload_list = (list(split(batch_list_keys, 60)))

#             def length_check(n):
#                 return len(n)

#             batch_upload_limits = list(map(length_check, batch_upload_list))
#             print(batch_upload_limits)
#             batch_output_list = []
#             batch_limit = 0
#             for img_filename, img_df in self.dataset.df.groupby('img_filename'):
#                 img_path = str(
#                     PurePath(self.dataset.path_to_annotations, str(img_df.iloc[0].img_folder), img_filename))
#                 assert exists(img_path), f"File does not exist: {img_path}"
#                 # Create a region object for each bounding box in the dataset
#                 regions = []
#                 for index, row in img_df.iterrows():
#                     # Normalize the bounding box coordinates between 0 and 1
#                     x = Decimal(row.ann_bbox_xmin / row.img_width).min(1)
#                     y = Decimal(row.ann_bbox_ymin / row.img_height).min(1)
#                     w = Decimal(row.ann_bbox_width / row.img_width).min(1 - x)
#                     h = Decimal(row.ann_bbox_height / row.img_height).min(1 - y)

#                     regions.append(Region(
#                         tag_id=self.tags[row.cat_name].id,
#                         left=x,
#                         top=y,
#                         width=w,
#                         height=h
#                     )
#                     )
#                 with open(img_path, mode="rb") as image_contents:
#                     image_and_annotations = ImageFileCreateEntry(name=img_filename,
#                                                                  contents=image_contents.read(),
#                                                                  regions=regions)
#                     batch_output_list.append(image_and_annotations)

#                 if len(batch_output_list) == batch_upload_limits[batch_limit]:
#                     print(batch_upload_limits[batch_limit])
#                     retry_limit = 0
#                     while retry_limit < 5:
#                         try:
#                             upload_status = "Completed"
#                             upload_result = self.trainer.create_images_from_files(
#                                 self.project.id,
#                                 ImageFileCreateBatch(images=batch_output_list)
#                             )
#                         except Exception as e:
#                             print(e)
#                             self.custom_logger.exception(e)
#                             print("No. of retry count :", retry_limit)
#                             self.custom_logger.info(f"No. of retry count : {retry_limit}")
#                             print('Sleeping for 120 seconds')
#                             self.custom_logger.info('Sleeping for 120 seconds')
#                             time.sleep(120)
#                             retry_limit += 1
#                             continue
#                         break
#                     else:
#                         upload_status = "Aborted"
#                         self.custom_logger.info(f"Upload {upload_status} ....")
#                         print(f"Upload {upload_status} ....")
#                         sys.exit(1)
#                     time.sleep(0.2)
#                     # If upload is not successful, print details about that image for debugging
#                     image_status_dict = dict();
#                     if not upload_result.is_batch_successful:
#                         print("Image upload failed.")
#                         self.custom_logger.info("Image upload failed.")
#                         for image in upload_result.images:
#                             image_status_dict[image.source_url] = image.status
#                     else:
#                         print("Image upload completed.")
#                         self.custom_logger.info("Image upload completed.")
#                         for image in upload_result.images:
#                             image_status_dict[image.source_url] = image.status
#                     for image_key, status_value in image_status_dict.items():
#                         if status_value == "OK":
#                             try:
#                                 shutil.move(f"training_data/Images/items/{image_key}", "training_data_complete/")
#                                 general_name = os.path.splitext(image_key)[0]
#                                 shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_complete/")
#                             except Exception as e:
#                                 print(e)
#                                 self.custom_logger.exception(e)
#                         else:
#                             try:
#                                 shutil.move(f"training_data/Images/items/{image_key}", "training_data_failed/")
#                                 general_name = os.path.splitext(image_key)[0]
#                                 shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_failed/")
#                                 print(image_key, status_value)
#                                 self.custom_logger.info(f"{image_key}: {status_value}")
#                             except Exception as e:
#                                 print(e)
#                                 self.custom_logger.exception(e)
#                     batch_output_list = []
#                     batch_limit += 1
#                     print("Upload list refreshed")
#                 else:
#                     continue
#             # This will take a few minutes
#             print(f"Upload {upload_status} ....")
#             self.custom_logger.info(f"Upload {upload_status} ....")
#             t2 = time.localtime()
#             current_time_2 = time.strftime("%H:%M:%S", t2)
#             print("Upload End Time :", current_time_2)
#             self.custom_logger.info(f"Upload End Time :{current_time_2}")

#         # Image vise continuous Upload mode
#         else:
#             # Image vise continuous Upload mode
#             print("Image vise continuous Upload mode")
#             self.custom_logger.info("Image vise continuous Upload mode")
#             upload_status = "Completed"
#             t1 = time.localtime()
#             current_time_1 = time.strftime("%H:%M:%S", t1)
#             print("Upload Start Time :", current_time_1)
#             self.custom_logger.info(f"Upload Start Time : {current_time_1}")
#             print("Upload Started ....")
#             self.custom_logger.info("Upload Started ....")
#             # Checking data set information
#             for img_filename, img_df in self.dataset.df.groupby('img_filename'):
#                 img_path = str(
#                     PurePath(self.dataset.path_to_annotations, str(img_df.iloc[0].img_folder), img_filename))
#                 assert exists(img_path), f"File does not exist: {img_path}"
#                 # Create a region object for each bounding box in the dataset
#                 regions = []
#                 for index, row in img_df.iterrows():
#                     # Normalize the bounding box coordinates between 0 and 1
#                     x = Decimal(row.ann_bbox_xmin / row.img_width).min(1)
#                     y = Decimal(row.ann_bbox_ymin / row.img_height).min(1)
#                     w = Decimal(row.ann_bbox_width / row.img_width).min(1 - x)
#                     h = Decimal(row.ann_bbox_height / row.img_height).min(1 - y)

#                     regions.append(Region(
#                         tag_id=self.tags[row.cat_name].id,
#                         left=x,
#                         top=y,
#                         width=w,
#                         height=h
#                     )
#                     )

#                 # Create an object with the image and all of the annotations for that image
#                 with open(img_path, mode="rb") as image_contents:
#                     image_and_annotations = [
#                         ImageFileCreateEntry(name=img_filename, contents=image_contents.read(),
#                                              regions=regions)]
#                 # Upload the image and all annotations for that image
#                 retry_limit = 0
#                 while retry_limit < 5:
#                     try:
#                         upload_status = "Completed"
#                         upload_result = self.trainer.create_images_from_files(
#                             self.project.id,
#                             ImageFileCreateBatch(images=image_and_annotations)
#                         )
#                     except Exception as e:
#                         print(e)
#                         self.custom_logger.exception(e)
#                         print("No. of retry count :", retry_limit)
#                         self.custom_logger.info(f"No. of retry count : {retry_limit}")
#                         print('Sleeping for 120 seconds')
#                         self.custom_logger.info('Sleeping for 120 seconds')
#                         time.sleep(120)
#                         retry_limit += 1
#                         continue
#                     break
#                 else:
#                     upload_status = "Aborted"
#                     self.custom_logger.info(f"Upload {upload_status} ....")
#                     print(f"Upload {upload_status} ....")
#                     sys.exit(1)
#                 time.sleep(0.2)
#                 # If upload is not successful, print details about that image for debugging
#                 image_status_dict = dict();
#                 if not upload_result.is_batch_successful:
#                     print("Image upload failed.")
#                     self.custom_logger.info("Image upload failed.")
#                     for image in upload_result.images:
#                         image_status_dict[image.source_url] = image.status
#                 else:
#                     print("Image upload completed.")
#                     self.custom_logger.info("Image upload completed.")
#                     for image in upload_result.images:
#                         image_status_dict[image.source_url] = image.status
#                 for image_key, status_value in image_status_dict.items():
#                     if status_value == "OK":
#                         try:
#                             shutil.move(f"training_data/Images/items/{image_key}", "training_data_complete/")
#                             general_name = os.path.splitext(image_key)[0]
#                             shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_complete/")
#                         except Exception as e:
#                             print(e)
#                             self.custom_logger.exception(e)
#                     else:
#                         try:
#                             shutil.move(f"training_data/Images/items/{image_key}", "training_data_failed/")
#                             general_name = os.path.splitext(image_key)[0]
#                             shutil.move(f"training_data/labels/yolo/{general_name}.txt", "training_data_failed/")
#                             print(image_key, status_value)
#                             self.custom_logger.info(f"{image_key} : {status_value}")
#                         except Exception as e:
#                             print(e)
#                             self.custom_logger.exception(e)
#             # This will take a few minutes
#             print("Upload {} ....".format(upload_status))
#             self.custom_logger.info(f"Upload {upload_status} ....")
#             t2 = time.localtime()
#             current_time_2 = time.strftime("%H:%M:%S", t2)
#             print("Upload End Time :", current_time_2)
#             self.custom_logger.info(f"Upload End Time :{current_time_2}")

#         print("Training of Model Started ....")
#         self.custom_logger.info("Training of Model Started ....")
#         try:
#             self.iteration = self.trainer.train_project(project_id=self.project.id, training_type=self.training_type,
#                                                         reserved_budget_in_hours=self.reserved_budget_in_hours,
#                                                         notification_email_address=self.notification_email)
#             while self.iteration.status != "Completed":
#                 self.iteration = self.trainer.get_iteration(self.project.id, self.iteration.id)
#                 t = time.localtime()
#                 current_time = time.strftime("%H:%M:%S", t)
#                 print('\r', "Training .... " + " " + self.project_name + " " + str(current_time), end='')
#                 time.sleep(1)
#             train_status = "Completed"
#         except Exception as e:
#             train_status = "Aborted"
#             print(e)
#             self.custom_logger.exception(e)

#         print("Training of Model {} ....".format(train_status))
#         self.custom_logger.info(f"Training of Model {train_status} ....")
#         # get performance data
#         try:
#             self.performance = self.trainer.get_iteration_performance(self.project.id, self.iteration.id,
#                                                                       threshold=self.probability_threshold_data,
#                                                                       overlap_threshold=self.overlap_threshold_data)
#             performance_dict = self.performance.as_dict()  # get performance data as dict
#             # print(performance_dict)
#             self.tag_performance_data = performance_dict.get("per_tag_performance")
#             self.overall_performance = dict();
#             self.overall_performance['Project Id'] = self.project.id
#             self.overall_performance['Iteration Id'] = self.iteration.id
#             self.overall_performance['Precision'] = performance_dict.get('precision')
#             self.overall_performance['Recall'] = performance_dict.get('recall')
#             self.overall_performance['MAP'] = performance_dict.get('average_precision')
#             self.overall_performance['Performance per Tag'] = self.tag_performance_data

#             print(self.overall_performance)
#             self.custom_logger.info(f"{self.overall_performance}")

#         except Exception as e:
#             print("No new iterations generated")
#             self.custom_logger.exception(e)

#         # Export Model
#         try:
#             platform = self.platform_val
#             self.export = self.trainer.export_iteration(self.project.id, self.iteration.id, platform, raw=False)
#             while self.export.status == "Exporting":
#                 print('\r', "Waiting for export ...", end='')
#                 time.sleep(10)
#                 self.exports = self.trainer.get_exports(self.project.id, self.iteration.id)
#                 # Locate the export for this iteration and check its status
#                 for e in self.exports:
#                     if e.platform == self.export.platform and e.flavor == self.export.flavor:
#                         self.export = e
#                         break
#                 print("Export status is: ", self.export.status)
#                 self.custom_logger.info(f"Export status is: {self.export.status}")
#         except Exception as e:
#             print("No new iteration for Exporting")
#             print(e)
#             self.custom_logger.exception(e)
#         time.sleep(300)
#         # Downloading as zip
#         try:
#             if self.export.status == "Done":
#                 if self.endpoint_status == 'Private':
#                     # Success, now we can download it with Private endpoint
#                     # v3.4 is successes only fails with version 3.3
#                     ###### VERY IMP KEEP VERSION V3.4 IN THE URL ######
#                     output = f'{self.training_endpoint}customvision/v3.4-preview/training/projects/{self.project.id}/artifacts?path={self.export.download_uri}'
#                     # passing headers is very IMP
#                     export_file = requests.get(output, headers={'Training-Key': f'{self.training_key}'})
#                     print('Endpoint is private')
#                     self.custom_logger.info('Endpoint is private')
#                     print('Saving the model in Zip file')
#                     self.custom_logger.info('Saving the model in Zip file')
#                     with open(self.iteration.id + "_export.zip", "wb") as file:
#                         file.write(export_file.content)
#                         print(" Model exported to ", self.iteration.id + "_export.zip")
#                         self.custom_logger.info(f"Model exported to {self.iteration.id}_export.zip")
#                 else:
#                     # Success, now we can download it with Public Endpoint
#                     export_file = requests.get(self.export.download_uri)
#                     print('Endpoint is public')
#                     self.custom_logger.info('Endpoint is public')
#                     print('Saving the model in Zip file')
#                     self.custom_logger.info('Saving the model in Zip file')
#                     with open(self.iteration.id + "_export.zip", "wb") as file:
#                         file.write(export_file.content)
#                         print(" Model exported to ", self.iteration.id + "_export.zip")
#                         self.custom_logger.info(f"Model exported to {self.iteration.id}_export.zip")
#         except Exception as e:
#             print("No new iteration for Downloading")
#             self.custom_logger.exception(e)
#         # Unzip the exported folder
#         try:
#             print('Unzipping the model exported')
#             self.custom_logger.info('Unzipping the model exported')
#             with zipfile.ZipFile(self.iteration.id + "_export.zip", 'r') as zip_ref:
#                 zip_ref.extractall(self.iteration.id + "_export")
#             print('Successfully Unzipped the folder')
#             self.custom_logger.info('Successfully Unzipped the folder')
#         except Exception as e:
#             print("Cannot unzip the model folder")
#             self.custom_logger.exception(e)

#         # Copy model.onnx and labels file to model registry
#         try:
#             print('Coping model.onnx and labels.txt for deployment to model registry ')
#             self.custom_logger.info('Coping model.onnx and labels.txt for deployment to model registry')
#             os.chmod(self.iteration.id + "_export", stat.S_IRWXO)
#             shutil.copy(self.iteration.id + "_export" + "/model.onnx", f"model_registry/{self.iteration.id}model.onnx")
#             shutil.copy(self.iteration.id + "_export" + "/labels.txt", f"model_registry/{self.iteration.id}labels.txt")
#             print('Done Copying files to model registry')
#             self.custom_logger.info('Done Copying files to model registry')
#         except Exception as e:
#             print("Could not copy the files to to model registry")
#             self.custom_logger.exception(e)

#         # Publishing Model as REST API ENDPOINT TO BE USED FOR PREDICTION
#         # The iteration is now trained. Publish it to the project endpoint

#         try:
#             self.publish_iteration_name_final = self.publish_iteration_name + self.iteration.id
#             self.trainer.publish_iteration(self.project.id, self.iteration.id, self.publish_iteration_name_final,
#                                            self.prediction_resource_id)
#             print("Done Publishing Iteration as End Point!")
#             self.custom_logger.info("Done Publishing Iteration as End Point!")
#         except Exception as e:
#             print("No new iteration for Publishing")
#             self.custom_logger.exception(e)

#         # Training Central Repo Operations
#         # Copying data to Central Training Repo incrementally in and also adding new iteration data.
#         # Create new Repo for First iteration and incremental iteration old + incremental data
#         try:
#             directory = 'Central Repo Training Data'
#             try:
#                 last_iteration = max([os.path.join(directory, d) for d in os.listdir(directory)],
#                                      key=os.path.getmtime)
#             except Exception as e:
#                 last_iteration = None

#             new_folder = f'{directory}/{self.iteration.id}'
#             if last_iteration is not None:
#                 src = last_iteration.replace("\\", "/")
#                 dst = new_folder
#                 shutil.copytree(src, dst)
#                 print('Data transfer from last downloaded iteration for Training Central Repo')
#                 self.custom_logger.info('Data transfer from last downloaded iteration for Training Central Repo')
#             else:
#                 os.mkdir(new_folder)
#                 os.mkdir(new_folder + "/Success")
#                 os.mkdir(new_folder + "/Failure")
#                 print('No existing data so we are first iteration for Training Central Repo')
#                 self.custom_logger.info('No existing data so we are first iteration for Training Central Repo')
#             print(f'Adding Iteration data to Training Central Repo for iteration {self.iteration.id}')
#             self.custom_logger.info(f'Adding Iteration data to Training Central Repo for iteration {self.iteration.id}')
#             upload_complete_folder = os.listdir("training_data_complete")
#             upload_failure_folder = os.listdir("training_data_failed")

#             # Move Successful files to central repo
#             if len(upload_complete_folder) == 0:
#                 print("No new Successful Data for current iteration training")
#                 self.custom_logger.info("No new Successful Data for current iteration training")
#             else:
#                 for root_train, dirs_train, files_train in os.walk('training_data_complete'):
#                     for file_train_val in files_train:
#                         path_file = os.path.join(root_train, file_train_val)
#                         shutil.copy2(path_file, new_folder + "/Success")

#             # Move Failure files to central repo
#             if len(upload_failure_folder) == 0:
#                 print("No new Failure Data for current iteration training")
#                 self.custom_logger.info("No new Failure Data for current iteration training")
#             else:
#                 for root_train, dirs_train, files_train in os.walk('training_data_failed'):
#                     for file_train_val in files_train:
#                         path_file = os.path.join(root_train, file_train_val)
#                         shutil.copy2(path_file, new_folder + "/Failure")
#             print(f'Successfully Added Iteration data to Training Central Repo for iteration {self.iteration.id}')
#             self.custom_logger.info(
#                 f'Successfully Added Iteration data to Training Central Repo for iteration {self.iteration.id}')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         # Ground truth Central Repo Operations
#         # Copying data to Central Ground truth Repo incrementally in and also adding new iteration data.
#         # Create new Repo for First iteration and incremental iteration old + incremental data
#         try:
#             directory_gt = 'Central Repo Ground Truth Data'
#             self.iteration_count = (len(next(os.walk(directory_gt))[1])) + 1
#             print(self.iteration_count)
#             self.custom_logger.info(f"{self.iteration_count}")
#             try:
#                 last_iteration_gt = max([os.path.join(directory_gt, d) for d in os.listdir(directory_gt)],
#                                         key=os.path.getmtime)
#             except Exception as e:
#                 last_iteration_gt = None

#             self.new_folder_gt = f'{directory_gt}/{self.iteration.id}'
#             if last_iteration_gt is not None:
#                 src_gt = last_iteration_gt.replace("\\", "/")
#                 dst_gt = self.new_folder_gt
#                 print(src_gt)
#                 print(dst_gt)
#                 shutil.copytree(src_gt, dst_gt)
#                 print('Data transfer from last downloaded iteration for Ground Truth Central Repo')
#                 self.custom_logger.info('Data transfer from last downloaded iteration for Ground Truth Central Repo')
#             else:
#                 os.mkdir(self.new_folder_gt)
#                 os.mkdir(self.new_folder_gt + '/original Annotation')
#                 os.mkdir(self.new_folder_gt + '/original images')
#                 os.mkdir(self.new_folder_gt + '/original Json')
#                 print('No existing data so we are first iteration for Ground Truth Central Repo')
#                 self.custom_logger.info('No existing data so we are first iteration for Ground Truth Central Repo')
#             print(f'Adding Iteration data to Ground Truth Central Repo for iteration {self.iteration.id}')
#             self.custom_logger.info(
#                 f'Adding Iteration data to Ground Truth Central Repo for iteration {self.iteration.id}')

#             for root_gt, dirs_gt, files_gt in os.walk(self.test_image_location):
#                 for files_gt_val in files_gt:
#                     # Creating folders for placing data in right folders for Ground truth summary report
#                     if files_gt_val.endswith(('.png', '.jpg', '.jpeg')):
#                         path_file_gt = os.path.join(root_gt, files_gt_val)
#                         shutil.copy2(path_file_gt, self.new_folder_gt + '/original images')

#                     elif files_gt_val.endswith('.txt'):
#                         path_file_gt = os.path.join(root_gt, files_gt_val)
#                         shutil.copy2(path_file_gt, self.new_folder_gt + '/original Annotation')

#                     elif files_gt_val.endswith('.json'):
#                         path_file_gt = os.path.join(root_gt, files_gt_val)
#                         shutil.copy2(path_file_gt, self.new_folder_gt + '/original Json')
#                     else:
#                         pass
#             shutil.copy2(f'training_data/labels/{self.dataset_name}.names', self.new_folder_gt)

#             print(f'Successfully Added Iteration data to Ground Truth Central Repo for iteration {self.iteration.id}')
#             self.custom_logger.info(
#                 f'Successfully Added Iteration data to Ground Truth Central Repo for iteration {self.iteration.id}')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         # Ground truth testing
#         print(f" Ground truth testing for test data started with published endpoint mode for {self.new_folder_gt}")
#         self.custom_logger.info(
#             f" Ground truth testing for test data started with published endpoint mode for {self.new_folder_gt}")
#         print(f" Generating for {self.new_folder_gt}")
#         self.custom_logger.info(f" Generating for {self.new_folder_gt}")
#         try:
#             self.ground_truth_subfolders = []
#             # Creating folders for predicted txt files
#             if os.path.exists(self.new_folder_gt + '/predicted'):
#                 shutil.rmtree(self.new_folder_gt + '/predicted')
#             else:
#                 pass
#             os.mkdir(self.new_folder_gt + '/predicted')
#             for (root, dirs, files) in os.walk(self.new_folder_gt, topdown=True):
#                 root = root.replace("\\", "/")
#                 self.ground_truth_subfolders.append(root + '/')
#             # print(self.ground_truth_subfolders)
#             self.ground_truth_subfolders = self.ground_truth_subfolders[1:]
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         # Defining Probability Threshold
#         probability_threshold = 0.4
#         for ground_truth_folder in self.ground_truth_subfolders:
#             try:
#                 for file_name in [file for file in os.listdir(ground_truth_folder)]:
#                     if file_name.endswith(('.png', '.jpg', '.jpeg')):
#                         with open(ground_truth_folder + file_name, mode="rb") as img_file:
#                             results = self.predictor.detect_image(self.project.id,
#                                                                   self.publish_iteration_name_final,
#                                                                   img_file)
#                         prediction_list = []
#                         for prediction in results.predictions:
#                             data_dict = dict();
#                             if prediction.probability >= probability_threshold:
#                                 data_dict['tag_name'] = prediction.tag_name
#                                 data_dict['probability'] = prediction.probability
#                                 data_dict['bounding_box.left'] = prediction.bounding_box.left
#                                 data_dict['bounding_box.height'] = prediction.bounding_box.height
#                                 data_dict['bounding_box.top'] = prediction.bounding_box.top
#                                 data_dict['bounding_box.width'] = prediction.bounding_box.width
#                                 prediction_list.append(data_dict)
#                             else:
#                                 pass
#                         img_read_val = cv2.cvtColor(cv2.imread(ground_truth_folder + file_name), cv2.COLOR_BGR2RGB)
#                         dh, dw = img_read_val.shape[:2]
#                         # Predicted and original image bounding box
#                         for prediction_result in prediction_list:
#                             tag = self.tag_list.index(prediction_result['tag_name'])
#                             left = prediction_result['bounding_box.left']
#                             height = prediction_result['bounding_box.height']
#                             top = prediction_result['bounding_box.top']
#                             width = prediction_result['bounding_box.width']
#                             probability = prediction_result['probability']
#                             left_coordinate, top_coordinate = int(left * dw), int(top * dh)
#                             width_coordinate, height_coordinate = int(left_coordinate + (width * dw)), int(
#                                 top_coordinate + (height * dh))
#                             predicted_filename = os.path.splitext(file_name)
#                             with open(self.new_folder_gt + '/predicted/' + predicted_filename[
#                                 0] + '.txt',
#                                       mode="a+") as predicted:
#                                 val = f"{tag} {probability} {left_coordinate} {top_coordinate} " \
#                                       f"{width_coordinate} {height_coordinate}"
#                                 predicted.write(val + '\n')
#                     else:
#                         pass
#             except Exception as e:
#                 print(e)
#                 self.custom_logger.exception(e)
#                 continue

#         # New Ground truth Summary report logic
#         print(f" Generating Summary Ground truth Report for {self.new_folder_gt}")
#         self.custom_logger.info(f" Generating Summary Ground truth Report for {self.new_folder_gt}")
#         try:
#             from gt_inference import ground_truth_pred_main
#             ground_truth_pred_main(self.new_folder_gt, self.dataset_name, self.iteration_count,
#                                    self.gt_report_config_dict)
#             print(f" Generated Summary Ground truth Report for {self.new_folder_gt}")
#             self.custom_logger.info(f" Generated Summary Ground truth Report for {self.new_folder_gt}")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         # New Detail Ground truth Report logic
#         print(f" Generating Detail Ground truth Report for {self.new_folder_gt}")
#         self.custom_logger.info(f" Generating Detail Ground truth Report for {self.new_folder_gt}")
#         try:
#             if os.path.exists("updated_groundtruth_images"):
#                 shutil.rmtree("updated_groundtruth_images")
#             else:
#                 pass
#             os.mkdir('updated_groundtruth_images')
#             image_row = 1
#             # Creating Excel Workbook..
#             workbook = xlsxwriter.Workbook(self.new_folder_gt + '/Detailed Report.xlsx')
#             # Defining the Alignment required..
#             my_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': 'True'})
#             # Creating worksheet..
#             worksheet = workbook.add_worksheet('Ground Truth Validation Report')
#             worksheet_1 = workbook.add_worksheet('Iteration Report')
#             worksheet_1.set_column(0, 9, 20)
#             for index, performance_data_val in enumerate(self.overall_performance.items()):
#                 worksheet_1.write(0, index, performance_data_val[0], my_format)
#                 worksheet_1.write(1, index, str(performance_data_val[1]), my_format)
#             worksheet_1.write(0, 7, 'Details of Meta data for Training images Part - 1', my_format)
#             worksheet_1.write(1, 7, str(self.training_data_types_part1), my_format)
#             worksheet_1.write(0, 8, 'Details of Meta data for Training images Part - 2', my_format)
#             worksheet_1.write(1, 8, str(self.training_data_types_part2), my_format)
#             worksheet_1.write(0, 9, 'Details of Augmentation', my_format)
#             worksheet_1.write(1, 9, str(self.transforms), my_format)
#             # Resizing Excel Sheet's Rows and Columns to fit our Data
#             worksheet.set_column(0, 0, 10)
#             worksheet.set_column(1, 20, 50)
#             worksheet.set_column(2, 2, 55)
#             worksheet.set_default_row(266)
#             worksheet.set_row(0, 14.5)
#             # Defining Header Data..
#             worksheet.write(0, 0, 'Sr.No', my_format)
#             worksheet.write(0, 1, 'File Name', my_format)
#             worksheet.write(0, 2, 'Image', my_format)
#             worksheet.write(0, 3, 'Predicted Annotations', my_format)
#             worksheet.write(0, 4, 'Original Annotations', my_format)
#             worksheet.write(0, 5, 'IOU Threshold Annotations', my_format)
#             dir_list = os.listdir(self.new_folder_gt + '/original images')
#             for image_data in dir_list:
#                 if image_data.endswith(('.png', '.jpg', '.jpeg')):
#                     img_read_val_detail = cv2.cvtColor(
#                         cv2.imread(self.new_folder_gt + '/original images/' + image_data), cv2.COLOR_BGR2RGB)
#                     dh, dw = img_read_val_detail.shape[:2]
#                     list_of_info_detail = []
#                     original_annotation_detail = []
#                     try:
#                         file_txt_name = os.path.splitext(image_data)
#                         if os.path.exists(self.new_folder_gt + '/predicted/' + file_txt_name[0] + '.txt'):
#                             for line in open(self.new_folder_gt + '/predicted/' + file_txt_name[0] + '.txt'):
#                                 original_annotation_dict_detail = dict();
#                                 if line.rstrip('\n'):
#                                     label_pred, prob, left_pred, top_pred, width_pred, height_pred = [
#                                         int(element) if index != 1 else float(element) for
#                                         index, element in
#                                         enumerate(line.rstrip('\n').split(' '))]

#                                     list_of_info_detail.append([left_pred, top_pred, width_pred, height_pred])
#                                     original_annotation_dict_detail['tag_name'] = self.tag_list[label_pred]
#                                     original_annotation_dict_detail['bounding_box.left'] = left_pred
#                                     original_annotation_dict_detail['bounding_box.top'] = top_pred
#                                     original_annotation_dict_detail['bounding_box.width'] = width_pred
#                                     original_annotation_dict_detail['bounding_box.height'] = height_pred
#                                     original_annotation_detail.append(original_annotation_dict_detail)
#                             for original_result_detail in list_of_info_detail:
#                                 cv2.rectangle(img_read_val_detail,
#                                               (original_result_detail[0], original_result_detail[1]),
#                                               (original_result_detail[2], original_result_detail[3]),
#                                               (0, 0, 0), 6)
#                         else:
#                             original_annotation_detail = []
#                             pass
#                     except Exception as e:
#                         print(e)
#                         self.custom_logger.exception(e)

#                     list_of_info = []
#                     original_annotation = []
#                     try:
#                         file_txt_name = os.path.splitext(image_data)
#                         for line in open(self.new_folder_gt + '/original Annotation/' + file_txt_name[0] + '.txt'):
#                             original_annotation_dict = dict();
#                             if line.rstrip('\n'):
#                                 label, x, y, w, h = [float(element) if index > 0 else int(element) for
#                                                      index, element in
#                                                      enumerate(line.rstrip('\n').split(' '))]
#                                 l = int(max((x - w / 2) * dw, 0))
#                                 r = int(min((x + w / 2) * dw, dw - 1))
#                                 t = int(max((y - h / 2) * dh, 0))
#                                 b = int(min((y + h / 2) * dh, dh - 1))
#                                 list_of_info.append([l, t, r, b])
#                                 original_annotation_dict['tag_name'] = self.tag_list[label]
#                                 original_annotation_dict['bounding_box.left'] = l
#                                 original_annotation_dict['bounding_box.top'] = t
#                                 original_annotation_dict['bounding_box.width'] = r
#                                 original_annotation_dict['bounding_box.height'] = b
#                                 original_annotation.append(original_annotation_dict)
#                         for original_result in list_of_info:
#                             cv2.rectangle(img_read_val_detail, (original_result[0], original_result[1]),
#                                           (original_result[2], original_result[3]),
#                                           (0, 255, 0), 6)

#                     except Exception as e:
#                         print(e)
#                         self.custom_logger.exception(e)

#                     try:
#                         file_json_name = os.path.splitext(image_data)
#                         meta_data_ground_truth = get_dict_from_json(
#                             self.new_folder_gt + '/original Json/' + file_json_name[0] + '.json')
#                         meta_data_ground_truth_data = meta_data_ground_truth.get('user')
#                     except Exception as e:
#                         print(e)
#                         self.custom_logger.exception(e)

#                     img = cv2.resize(img_read_val_detail, (350, 350))
#                     cv2.imwrite('updated_groundtruth_images/' + image_data, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                     # Writing the Data in Excel Workbook
#                     worksheet.write(image_row, 0, image_row, my_format)
#                     worksheet.write(image_row, 1, image_data, my_format)

#                     worksheet.insert_image(image_row, 2, 'updated_groundtruth_images/' + image_data,
#                                            {'align': 'center', 'valign': 'vcenter'})
#                     worksheet.write(image_row, 4, str(original_annotation), my_format)
#                     iou_data = []
#                     if len(original_annotation_detail) == 0:
#                         worksheet.write(image_row, 3, 'No Prediction', my_format)
#                         worksheet.write(image_row, 5, 'No Prediction', my_format)
#                     else:
#                         worksheet.write(image_row, 3, str(original_annotation_detail), my_format)
#                         for annotation in original_annotation_detail:
#                             tag_val = annotation['tag_name']
#                             original_list = []
#                             for original_data in original_annotation:
#                                 if original_data['tag_name'] == tag_val:
#                                     original_list.append(original_data)
#                                 else:
#                                     pass
#                             if len(original_list) == 0:
#                                 pass
#                             else:
#                                 iou_val_prediction = []
#                                 for original_data_val in original_list:
#                                     annotation_dict_pred = annotation
#                                     annotation_dict_original = original_data_val
#                                     annotation_box_pred = [annotation_dict_pred['bounding_box.left'],
#                                                            annotation_dict_pred['bounding_box.top'],
#                                                            annotation_dict_pred['bounding_box.width'],
#                                                            annotation_dict_pred['bounding_box.height']
#                                                            ]
#                                     annotation_box_original = [annotation_dict_original['bounding_box.left'],
#                                                                annotation_dict_original['bounding_box.top'],
#                                                                annotation_dict_original['bounding_box.width'],
#                                                                annotation_dict_original['bounding_box.height']
#                                                                ]
#                                     box1 = torch.tensor([annotation_box_pred], dtype=torch.float)
#                                     box2 = torch.tensor([annotation_box_original], dtype=torch.float)
#                                     iou = bops.box_iou(box1, box2)
#                                     iou_percent = (iou * 100)
#                                     iou_val_prediction.append(int(iou_percent))
#                                 iou_data.append(max(iou_val_prediction))
#                     worksheet.write(image_row, 5, str(iou_data), my_format)
#                     col_val = 6
#                     for meta_data_key, meta_data_val in meta_data_ground_truth_data.items():
#                         worksheet.write(0, col_val, meta_data_key, my_format)
#                         worksheet.write(image_row, col_val, meta_data_val, my_format)
#                         col_val += 1
#                     image_row += 1
#                 else:
#                     pass
#             workbook.close()
#             if os.path.exists(self.new_folder_gt + '/Detailed Report.xlsx'):
#                 print('Ground Truth data Logged in Excel file for Human in Loop review')
#                 self.custom_logger.info('Ground Truth data Logged in Excel file for Human in Loop review')
#             else:
#                 print('Unable to Login Ground Truth data in Excel File ')
#                 self.custom_logger.info('Unable to Login Ground Truth data in Excel File ')

#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         print('Generating Latest Reports in gt_reports')
#         self.custom_logger.info('Generating Latest Reports in gt_reports')
#         try:
#             shutil.copy2(f"{self.new_folder_gt}/Detailed Report.xlsx", "gt_reports")
#             print('All Ground truth Latest Reports in gt_reports')
#             self.custom_logger.info('All Ground truth Latest Reports in gt_reports')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         # Log details to MLFLOW FOR TRACKING
#         try:
#             self.uuid_custom = uuid.uuid1()
#             mlflow.set_experiment(self.project_name)
#             mlflow.log_param('Project Name', self.project_name)
#             # mlflow.log_param('Details of Training images Part 1', self.training_data_types_part1)
#             # mlflow.log_param('Details of Training images Part 2', self.training_data_types_part2)
#             mlflow.log_param('Details of Augmentation', str(self.augmentation_of_data))
#             mlflow.log_param('Details of Special Augmentation', str(self.special_augmentation_of_data))
#             mlflow.log_param('Project Id', self.project.id)
#             mlflow.log_param('Iteration Id', self.iteration.id)
#             mlflow.log_param('Ground truth Details', self.new_folder_gt)
#             mlflow.log_param('Custom Iteration Id', self.uuid_custom)
#             mlflow.log_param('Probability Threshold', self.probability_threshold_data)
#             mlflow.log_param('Overlap Threshold', self.overlap_threshold_data)
#             mlflow.log_metric('Precision', self.overall_performance.get('Precision'))
#             mlflow.log_metric('Recall', self.overall_performance.get('Recall'))
#             mlflow.log_metric('MAP', self.overall_performance.get('MAP'))
#             mlflow.log_artifact('gt_reports/class_wise_report.xlsx')
#             mlflow.log_artifact('gt_reports/Detailed Report.xlsx')

#             for tag_data in self.tag_performance_data:
#                 tag_name = tag_data.get('name')
#                 tag_precision = tag_data.get('precision')
#                 tag_recall = tag_data.get('recall')
#                 tag_mAP = tag_data.get('average_precision')
#                 mlflow.log_metrics({tag_name + ' precision': tag_precision,
#                                     tag_name + ' recall': tag_recall, tag_name + ' MAP': tag_mAP})

#             print("Model details added to MLFLOW for tracking and tracing")
#             self.custom_logger.info("Model details added to MLFLOW for tracking and tracing")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#     def clean_resources(self):
#         # Delete all unwanted files
#         try:
#             os.remove(self.iteration.id + "_export.zip")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         try:
#             os.chmod(self.iteration.id + "_export", stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
#             shutil.rmtree(self.iteration.id + "_export")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         try:
#             os.chmod('training_data', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
#             shutil.rmtree('training_data')
#             shutil.rmtree('training_data_complete')
#             shutil.rmtree('training_data_failed')
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         try:
#             shutil.rmtree('updated_groundtruth_images')
#             shutil.rmtree("groundtruth_testing_data")
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)
#         try:
#             self.trainer.unpublish_iteration(self.project.id, self.iteration.id)
#         except Exception as e:
#             print(e)
#             self.custom_logger.exception(e)

#         print('Unpublished iteration and removed all unnecessary Files for storage optimization')
#         self.custom_logger.info('Unpublished iteration and removed all unnecessary Files for storage optimization')
#         try:
#             self.msg = ('Done With Azure Custom Vision Operations for Project Name : {} Iteration ID : {}'
#                         .format(self.project_name, self.iteration.id))
#         except Exception as e:
#             self.msg = ("No new Iterations performed for  Project Name : {}".format(self.project_name))

#         print(self.msg)
#         self.custom_logger.info(f"{self.msg}")
#         return self.msg


class AICloud:
    """ UPLOAD DATA TO MINIO FOR AI CLOUD """

    def __init__(self, ai_cloud_config_dict, dataloop_config_dict, custom_pipeline_yolo_version=None):
        self.ai_cloud_config_dict = ai_cloud_config_dict
        self.dataloop_config_dict = dataloop_config_dict
        self.custom_pipeline_yolo_version = custom_pipeline_yolo_version
        self.server = self.ai_cloud_config_dict.get('server')
        self.access_key = self.ai_cloud_config_dict.get('access_key')
        self.secret_key = self.ai_cloud_config_dict.get('secret_key')
        self.model_bucket = self.ai_cloud_config_dict.get('model_bucket')
        self.data_path = self.ai_cloud_config_dict.get('storage_data_path')
        self.training_data_path = self.ai_cloud_config_dict.get('training_data_path')
        self.training_data_folder = self.ai_cloud_config_dict.get('training_data_folder')
        self.dataset_name = self.dataloop_config_dict.get('dl_dataset_name').strip()
        self.tag_list = [line.strip() for line in open(f"training_data/labels/{self.dataset_name}.names", 'r')]
        self.target_model_path = None
        self.data_upload = None
        self.status = None

    def upload_data_to_minio(self):
        """ Correct Upload folder structure and upload to Minio after zip folder and cleanup resources """
        print("Creating / Updating obj.names and obj.data files as per Dataloop Dataset Recipe")
        try:
            try:
                # Auto updating of obj.data and obj.names file
                # Updating obj.data
                with open("AI Cloud Pipeline files/obj.data") as f:
                    lines = f.readlines()
                lines[0] = ('classes = {0}'.format(len(self.tag_list)) + '\n')
                with open("AI Cloud Pipeline files/obj.data", "w") as f:
                    f.writelines(lines)
                # Updating obj.names
                with open("AI Cloud Pipeline files/obj.names", 'w', encoding='utf-8') as f:
                    for labels in self.tag_list:
                        f.write(labels + '\n')
                print("Created / Updated obj.names and obj.data files as per Dataloop Dataset Recipe")
            except Exception as e:
                print(e)
            # Connect Minio Client
            print('Creating Connection with MinIO')
            self.data_path = self.data_path + '_' + str(datetime.datetime.now()) + '/'
            minio_client = Minio(self.server, access_key=self.access_key,
                                 secret_key=self.secret_key,
                                 secure=False)
            print('Rearranging data as per folder structure for AI Cloud Pipeline')
            # Generate Folders for Data Upload
            try:
                if os.path.exists("minio_training_data"):
                    shutil.rmtree("minio_training_data")
                else:
                    pass
                if os.path.exists("minio_ground_truth_data"):
                    shutil.rmtree("minio_ground_truth_data")
                else:
                    pass
                os.mkdir("minio_training_data")
                os.mkdir("minio_ground_truth_data")

                dir_src = ["training_data/", "groundtruth_testing_data/"]
                dir_dst = ["minio_training_data/", "minio_ground_truth_data"]
                for folder in range(len(dir_src)):
                    for root, _, files in os.walk(dir_src[folder]):
                        for file in files:
                            if file.endswith('.jpg') or file.endswith('.png') or file.endswith('.txt'):
                                copy(os.path.join(root, file), dir_dst[folder])
                            else:
                                pass
            except Exception as e:
                print(e)

            # Clean up resources
            try:
                os.chmod('training_data', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                shutil.rmtree('training_data')
            except Exception as e:
                print(e)
            try:
                shutil.rmtree("groundtruth_testing_data")
            except Exception as e:
                print(e)

            # Training data upload
            # Download old Master Training folder if it is present else create new one with folder structure
            print('Downloading / Creating Master Training Data from Minio')
            try:
                # Downloading Dataset
                target_model_path = self.training_data_path + self.training_data_folder + '.zip'
                minio_client.fget_object(self.model_bucket, target_model_path, self.training_data_folder + '.zip')
                print('Downloaded Master Training Data from Minio')
                # Unzipping Folder
                with zipfile.ZipFile(self.training_data_folder + ".zip", "r") as zip_ref:
                    zip_ref.extractall(self.training_data_folder)
                print('Unzipped Master Training Data Downloaded from Minio')
            except Exception as e:
                print('Creating Master Training Data')
                os.mkdir(self.training_data_folder)
                os.mkdir(self.training_data_folder + '/' + 'cfg')
                os.mkdir(self.training_data_folder + '/' + 'data')
                os.mkdir(self.training_data_folder + '/' + 'data' + '/' + 'obj')
                print('Created Master Training Data')

            # Copying all necessary files and incremental data to Master Training folder before upload
            try:
                print('Adding Incremental data and config and txt files for Master Training Data')
                # config file to cfg
                copy_tree("AI Cloud framework_config", self.training_data_folder + '/' + 'cfg')
                # other files to data
                copy_tree("AI Cloud Pipeline files", self.training_data_folder + '/' + 'data')
                # Incremental data to data/obj
                copy_tree("minio_training_data", self.training_data_folder + '/' + 'data' + '/' + 'obj')
                print('Generating train.txt file')
                # Generating train.txt
                ext = ['png', 'jpg', 'tiff']
                files = []
                [files.extend(glob.glob(self.training_data_folder + '/' + 'data' + '/' + 'obj/'
                                        + '*.' + e)) for e in ext]
                with open(self.training_data_folder + '/' + 'data' + "/train.txt", 'w', encoding='utf-8') as f:
                    for img in files:
                        img = img.replace("\\", "/")
                        f.write(img + '\n')
                print('Successfully generated train.txt file')
            except Exception as e:
                print(e)

            # Backup Data and training data Upload
            # Zipping Files to be uploaded
            print('Zipping Folders to be Uploaded to Minio')
            try:
                shutil.make_archive(self.training_data_folder, 'zip', self.training_data_folder)
                shutil.make_archive('minio_training_data', 'zip', 'minio_training_data')
                shutil.make_archive('minio_ground_truth_data', 'zip', 'minio_ground_truth_data')
            except Exception as e:
                print(e)
            print('Zipped Folders to be Uploaded to Minio')

            # Upload Backup data and Master Training to Minio
            print('Started Uploading Master Training Data and Storage Data to Minio')
            try:
                self.target_model_path = [self.data_path + 'minio_training_data.zip',
                                          self.data_path + 'minio_ground_truth_data.zip',
                                          self.training_data_path + self.training_data_folder + '.zip']
                self.data_upload = ['minio_training_data.zip',
                                    'minio_ground_truth_data.zip',
                                    self.training_data_folder + '.zip']
                for folder_data in range(len(self.target_model_path)):
                    with open(self.data_upload[folder_data], 'rb') as file_data:
                        file_stat = os.stat(self.data_upload[folder_data])
                        minio_client.put_object(self.model_bucket, self.target_model_path[folder_data],
                                                file_data, file_stat.st_size)
                print('Successfully Uploaded Data to Minio')
            except Exception as e:
                print(e)

            # Clean up resources once uploaded
            try:
                print('Started Clean up for existing resources')
                shutil.rmtree("minio_ground_truth_data")
                shutil.rmtree("minio_training_data")
                shutil.rmtree(self.training_data_folder)
                os.remove("minio_ground_truth_data.zip")
                os.remove("minio_training_data.zip")
                os.remove(self.training_data_folder + '.zip')
                print('Resources Cleaned up Successfully')
            except Exception as e:
                print(e)
            self.status = "Data Upload Successfully To Minio in desired folder structure with all necessary files"

        except Exception as e:
            self.status = "Failed to Upload Data To Minio"

        print(self.status)
        return self.status


class CustomYoloPipelineYoloV4:
    """ Custom Training Pipeline with Yolo model """

    def __init__(self, custom_config_dict, custom_logger, gt_report_config_dict, custom_pipeline_yolo_version):
        self.custom_logger = custom_logger
        self.custom_config_dict = custom_config_dict
        self.gt_report_config_dict = gt_report_config_dict
        self.custom_pipeline_yolo_version = custom_pipeline_yolo_version
        self.workers = self.custom_config_dict.get("workers")
        self.device = self.custom_config_dict.get("device")
        self.batch_size = self.custom_config_dict.get("batch_size")
        self.data_yaml = self.custom_config_dict.get("data_yaml")
        self.cfg_yaml = self.custom_config_dict.get("cfg_yaml")
        self.weights = self.custom_config_dict.get("weights")
        self.model_name = self.custom_config_dict.get("model_name")
        self.hyper_parameter_file = self.custom_config_dict.get("hyper_parameter_file")
        self.dataset_name = self.custom_config_dict.get("labels_file_name")
        self.tag_list = [line.strip() for line in open(f"training_data/labels/{self.dataset_name}.names", 'r')]
        self.augmentation_of_data = self.custom_config_dict.get("augmentation_of_data")
        self.augmentation_of_data_type = self.custom_config_dict.get("augmentation_of_data_type")
        self.special_augmentation_of_data = self.custom_config_dict.get("special_augmentation_of_data")
        self.transforms = None
        self.special_transforms = None
        self.length_augmentation_list = None
        self.labels_file_path = None
        
        # YOLOv8 specific parameters
        if self.custom_pipeline_yolo_version == "YoloV8":
            self.epochs = self.custom_config_dict.get("epochs", 100)
            self.imgsz = self.custom_config_dict.get("imgsz", 640)
            self.model_size = self.custom_config_dict.get("model_size", "yolov8n.pt")
            self.patience = self.custom_config_dict.get("patience", 50)
            self.save_period = self.custom_config_dict.get("save_period", 10)
            self.auto_augment = self.custom_config_dict.get("auto_augment","none")
            self.fliplr=self.custom_config_dict.get("fliplr", 0.0)

    def _parse_training_metrics(self, training_output):
        """Parse training metrics from YOLOv4 training output"""
        metrics = {}
        try:
            lines = training_output.split('\n')
            last_loss = None
            iteration_count = 0
            
            for line in lines:
                # Parse loss values - capture final loss
                if 'avg_loss' in line:
                    avg_loss_match = re.search(r'avg_loss = ([\d.]+)', line)
                    if avg_loss_match:
                        last_loss = float(avg_loss_match.group(1))
                
                # Parse learning rate - capture final learning rate
                if 'rate=' in line:
                    lr_match = re.search(r'rate=([\d.e-]+)', line)
                    if lr_match:
                        metrics['learning_rate'] = float(lr_match.group(1))
                
                # Count iterations
                if 'Loaded' in line and 'iterations' in line:
                    iter_match = re.search(r'(\d+) iterations', line)
                    if iter_match:
                        iteration_count = int(iter_match.group(1))
                
                # Parse iteration info from weight saving
                if 'Saving weights to' in line and 'weights' in line:
                    weights_match = re.search(r'_(\d+)\.weights', line)
                    if weights_match:
                        metrics['final_iteration'] = int(weights_match.group(1))
                
                # Parse mAP if available
                if 'mean_average_precision' in line or 'mAP' in line:
                    map_match = re.search(r'mAP.*?= ([\d.]+)', line)
                    if map_match:
                        metrics['mAP'] = float(map_match.group(1))
                
                # Parse training time information
                if 'Training stopped' in line or 'Total time' in line:
                    time_match = re.search(r'(\d+\.?\d*) seconds', line)
                    if time_match:
                        metrics['training_time_seconds'] = float(time_match.group(1))
            
            # Add final loss and iteration count
            if last_loss is not None:
                metrics['final_avg_loss'] = last_loss
            if iteration_count > 0:
                metrics['total_iterations'] = iteration_count
                        
        except Exception as e:
            print(f"Error parsing training metrics: {e}")
            
        return metrics

    def _get_gpu_metrics(self):
        """Get comprehensive GPU metrics using multiple methods"""
        gpu_metrics = {}
        
        # Method 1: Try GPUtil (simpler, widely compatible)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                for i, gpu in enumerate(gpus):
                    gpu_metrics[f'GPU_{i}_Usage_Percent'] = gpu.load * 100
                    gpu_metrics[f'GPU_{i}_Memory_Usage_Percent'] = gpu.memoryUtil * 100
                    gpu_metrics[f'GPU_{i}_Memory_Used_MB'] = gpu.memoryUsed
                    gpu_metrics[f'GPU_{i}_Memory_Total_MB'] = gpu.memoryTotal
                    gpu_metrics[f'GPU_{i}_Memory_Free_MB'] = gpu.memoryFree
                    gpu_metrics[f'GPU_{i}_Temperature_C'] = gpu.temperature
                    gpu_metrics[f'GPU_{i}_Name'] = gpu.name
                    gpu_metrics[f'GPU_{i}_UUID'] = gpu.uuid
                return gpu_metrics, f"GPUtil: Found {len(gpus)} GPU(s)"
        except ImportError:
            pass
        except Exception as e:
            print(f"GPUtil error: {e}")
        
        # Method 2: Try nvidia-ml-py3 (more detailed NVIDIA GPU info)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU name and info
                gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                gpu_metrics[f'GPU_{i}_Name'] = gpu_name
                
                # Memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_metrics[f'GPU_{i}_Memory_Used_MB'] = mem_info.used // (1024 * 1024)
                gpu_metrics[f'GPU_{i}_Memory_Total_MB'] = mem_info.total // (1024 * 1024)
                gpu_metrics[f'GPU_{i}_Memory_Free_MB'] = mem_info.free // (1024 * 1024)
                gpu_metrics[f'GPU_{i}_Memory_Usage_Percent'] = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_metrics[f'GPU_{i}_Usage_Percent'] = util.gpu
                    gpu_metrics[f'GPU_{i}_Memory_Util_Percent'] = util.memory
                except:
                    pass
                
                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_metrics[f'GPU_{i}_Temperature_C'] = temp
                except:
                    pass
                
                # Power usage
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000  # Convert to watts
                    gpu_metrics[f'GPU_{i}_Power_Usage_W'] = power
                    
                    # Power limits
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] // 1000
                    gpu_metrics[f'GPU_{i}_Power_Limit_W'] = power_limit
                except:
                    pass
                
                # Clock speeds
                try:
                    gpu_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                    gpu_metrics[f'GPU_{i}_Graphics_Clock_MHz'] = gpu_clock
                    gpu_metrics[f'GPU_{i}_Memory_Clock_MHz'] = mem_clock
                except:
                    pass
            
            pynvml.nvmlShutdown()
            return gpu_metrics, f"pynvml: Found {device_count} NVIDIA GPU(s)"
            
        except ImportError:
            pass
        except Exception as e:
            print(f"pynvml error: {e}")
        
        # Method 3: Try PyTorch CUDA if available
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_metrics[f'GPU_{i}_Name'] = gpu_name
                    
                    # Memory info from PyTorch
                    mem_allocated = torch.cuda.memory_allocated(i) // (1024 * 1024)  # MB
                    mem_cached = torch.cuda.memory_reserved(i) // (1024 * 1024)  # MB
                    
                    gpu_metrics[f'GPU_{i}_Memory_Allocated_MB'] = mem_allocated
                    gpu_metrics[f'GPU_{i}_Memory_Cached_MB'] = mem_cached
                
                return gpu_metrics, f"PyTorch CUDA: Found {device_count} GPU(s)"
        except ImportError:
            pass
        except Exception as e:
            print(f"PyTorch CUDA error: {e}")
        
        return {}, "No GPU monitoring libraries available or no GPUs detected"

    def perform_augmentation(self):

        """ Perform Special Augmentation on user provided classes and generic Augmentation on all classes"""
        print("Current Iteration has following classes : ", self.tag_list)
        print()
        special_transforms_class = input("Please Enter Classes from above list only on which you want to perform "
                                         "special Augmentation on with ',' "
                                         "separated values  (Input is Case sensitive) or NA if you want to perform "
                                         "None : ")
        try:
            special_class_index_list = []
            if special_transforms_class:
                special_transforms_class_list = list(special_transforms_class.split(","))
                # Getting index for all labels provided by user
                for special_class in special_transforms_class_list:
                    if special_class in self.tag_list:
                        special_class_index_list.append(self.tag_list.index(special_class))
                    else:
                        pass
                special_class_index_string = map(str, special_class_index_list)
                special_class_index_list = list(special_class_index_string)
                print("Index for which special Augmentation to be performed", special_class_index_list)
            else:
                special_class_index_list = []

        except Exception as e:
            special_class_index_list = []

        print(special_class_index_list)

        def write_image_box(image_write, bboxes_write, filename_write):
            """ Generate Augmented images and text files"""
            cv2.imwrite("training_data/Images/items/" + filename_write + ".jpg", image_write[:, :, ::-1])
            with open("training_data/labels/yolo/" + filename_write + ".txt", 'w+') as file:
                for box in bboxes_write:
                    x, y, w, h, category = box
                    file.write(f"{category} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            file.close()

        # IMPORT ONLY IN YOLO FORMAT

        # SPECIAL AUGMENTATION on User specific classes
        try:
            # Special data Augmentations performed or not on user specified class
            # See a list of transforms performed by Albumentations.
            # https://albumentations.ai/docs/getting_started/transforms_and_targets/
            if self.special_augmentation_of_data and special_class_index_list:
                print('Performing Special Augmentations on classes specified by the User')
                self.custom_logger.info('Performing Special Augmentations on classes specified by the User')
                self.special_transforms = []
                for special_aug_item in self.special_augmentation_of_data:
                    for special_aug_key, special_aug_val in special_aug_item.items():
                        special_aug_val["always_apply"] = True
                        special_aug_data = [getattr(A, special_aug_key)(**special_aug_val)]
                        self.special_transforms.append(special_aug_data)
                print("Special Aug list", self.special_transforms)
                self.custom_logger.info(f"Special Aug list {self.special_transforms}")

                self.labels_file_path = glob.glob(r'training_data/labels/yolo/*.txt')

                special_aug_files = []

                for txt_file in self.labels_file_path:
                    with open(txt_file) as fp:
                        first_index = zip(*[line.rstrip().split(' ') for line in fp])
                        first_list = list(first_index)
                        first_element = first_list[0]
                        valid_files = any(i in first_element for i in special_class_index_list)
                        if valid_files:
                            special_aug_files.append(txt_file)
                        else:
                            pass

                def do_split(file_path_data):
                    """ Perform split for filenames """
                    file_val = os.path.basename(file_path_data)
                    return os.path.splitext(file_val)[0]

                # print(special_aug_files)
                special_aug_images = list(map(do_split, special_aug_files))
                # print(special_aug_images)

                for special_images in special_aug_images:
                    try:
                        all_files = os.listdir("training_data/Images/items")
                        r = re.compile(f"{special_images}.(png|jpg)")
                        mapped_img = list(filter(r.match, all_files))
                        if mapped_img:
                            working_img = mapped_img[0]
                            special_filename = os.path.splitext(working_img)[0]
                            special_image = cv2.imread("training_data/Images/items/" + working_img)
                            special_image = cv2.cvtColor(special_image, cv2.COLOR_BGR2RGB)
                            annotation_path = "training_data/labels/yolo/" + str(special_filename) + ".txt"
                            lines = [line.rstrip('\n') for line in open(annotation_path)]
                            bboxes = []
                            if lines != ['']:
                                for line in lines:
                                    category, x, y, w, h = line.split(" ")
                                    bboxes.append([float(x), float(y), float(w), float(h), int(category)])

                            for special_index, special_aug in enumerate(self.special_transforms):
                                special_transforms = A.Compose(special_aug,
                                                               bbox_params=A.BboxParams(format="yolo", min_area=1,
                                                                                        min_visibility=0.5))
                                special_res = special_transforms(image=special_image, bboxes=bboxes)
                                special_filename_res = special_filename + "_special_" + str(special_index)
                                write_image_box(special_res["image"], special_res["bboxes"], special_filename_res)
                        else:
                            pass
                    except Exception as e:
                        print(e)
                        self.custom_logger.exception(e)
                        continue
                print('Successfully Performed Special Augmentations on user specified classes')
                self.custom_logger.info('Successfully Performed Special Augmentations on user specified classes')
            else:
                print('NO Special Augmentations performed')
                self.custom_logger.info('NO Special Augmentations performed')
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        def split_image_data(a, n):
            k, m = divmod(len(a), n)
            return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

        # GENERIC AUGMENTATION on all classes or semi on all classes
        try:
            # Standard Generic data Augmentations performed on all class or semi on all classes
            # See a list of transforms performed by Albumentations.
            # https://albumentations.ai/docs/getting_started/transforms_and_targets/
            if self.augmentation_of_data:
                self.transforms = []
                for aug_item in self.augmentation_of_data:
                    for aug_key, aug_val in aug_item.items():
                        aug_val["always_apply"] = True
                        aug_data = [getattr(A, aug_key)(**aug_val)]
                        self.transforms.append(aug_data)
                print("Generic Aug list", self.transforms)
                self.custom_logger.info(f"Generic Aug list{self.transforms}")

                if self.augmentation_of_data_type == "Full":
                    print('Performing Generic Standard Augmentations on all classes  as specified by the User')
                    self.custom_logger.info('Performing Generic Standard Augmentations on all classes  as specified by '
                                            'the User')
                    # Give each aug as list separated as given below 4 augmentation are performed

                    for file in os.listdir("training_data/Images/items"):
                        try:
                            if file.endswith(('.png', '.jpg', '.jpeg')):
                                filename = os.path.splitext(file)[0]
                                # print(os.path.abspath(file))
                                image = cv2.imread("training_data/Images/items/" + file)
                                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                annotation_path = "training_data/labels/yolo/" + str(filename) + ".txt"
                                lines = [line.rstrip('\n') for line in open(annotation_path)]
                                bboxes = []
                                if lines != ['']:
                                    for line in lines:
                                        category, x, y, w, h = line.split(" ")
                                        bboxes.append([float(x), float(y), float(w), float(h), int(category)])

                                for index, aug in enumerate(self.transforms):
                                    transforms = A.Compose(aug,
                                                           bbox_params=A.BboxParams(format="yolo", min_area=1,
                                                                                    min_visibility=0.5))
                                    res = transforms(image=image, bboxes=bboxes)
                                    filename_res = filename + "_" + str(index)
                                    write_image_box(res["image"], res["bboxes"], filename_res)
                            else:
                                pass
                        except Exception as e:
                            print(e)
                            self.custom_logger.exception(e)
                            continue
                    print('Successfully Performed Standard Generic Augmentations on all classes as specified by the '
                          'User')
                    self.custom_logger.info('Successfully Performed Standard Generic Augmentations on all classes as '
                                            'specified by the User')
                else:
                    print('Performing Generic Standard Augmentations on parts of classes only to have reduced data '
                          'volume')
                    self.custom_logger.info(
                        'Performing Generic Standard Augmentations on parts of classes only to have reduced data '
                        'volume')
                    self.length_augmentation_list = len(self.transforms)
                    self.labels_file_path = glob.glob(r'training_data/labels/yolo/*.txt')
                    for list_index in range(len(self.tag_list)):
                        print(f"Augmentation started for class {self.tag_list[list_index]}")
                        self.custom_logger.info(f"Augmentation started for class {self.tag_list[list_index]}")
                        images_list = []
                        for txt_file in self.labels_file_path:
                            txt_filename = os.path.basename(txt_file)
                            with open(txt_file) as fp:
                                first_index = zip(*[line.rstrip().split(' ') for line in fp])
                                first_list = list(first_index)
                                first_element = first_list[0]
                                # print(first_element)
                                if str(list_index) in first_element:
                                    images_list.append(txt_filename)
                                else:
                                    pass
                        print(f"Img list length class vise{len(images_list)}")
                        self.custom_logger.info(f"Img list length class vise{len(images_list)}")
                        random.shuffle(images_list)
                        augmentation_list_class_vise = (list(split_image_data(images_list,
                                                                              self.length_augmentation_list)))
                        print(f"augmentation_list_class_vise {augmentation_list_class_vise}")
                        self.custom_logger.info(f"augmentation_list_class_vise {augmentation_list_class_vise}")
                        for image_chunk_count, image_chunk_list in enumerate(augmentation_list_class_vise):
                            print(f"Augmentation performed {self.transforms[image_chunk_count]}")
                            self.custom_logger.info(f"Augmentation performed {self.transforms[image_chunk_count]}")
                            for image_file in image_chunk_list:
                                image_filename = os.path.splitext(image_file)[0]
                                all_files = os.listdir("training_data/Images/items")
                                r = re.compile(f"{image_filename}.(png|jpg)")
                                mapped_img = list(filter(r.match, all_files))
                                if mapped_img:
                                    try:
                                        image = cv2.imread(f"training_data/Images/items/{mapped_img[0]}")
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                        annotation_path = "training_data/labels/yolo/" + str(image_filename) + ".txt"
                                        lines = [line.rstrip('\n') for line in open(annotation_path)]
                                        bboxes = []
                                        if lines != ['']:
                                            for line in lines:
                                                category, x, y, w, h = line.split(" ")
                                                bboxes.append([float(x), float(y), float(w), float(h), int(category)])

                                        transforms = A.Compose(self.transforms[image_chunk_count],
                                                               bbox_params=A.BboxParams(format="yolo", min_area=1,
                                                                                        min_visibility=0.5))
                                        res = transforms(image=image, bboxes=bboxes)
                                        filename_res = image_filename + "_" + str(image_chunk_count)
                                        write_image_box(res["image"], res["bboxes"], filename_res)
                                    except Exception as e:
                                        print(e)
                                        self.custom_logger.exception(e)
                                        continue
                                else:
                                    print(f"No Image with name exist {mapped_img[0]}")
                                    self.custom_logger.info(f"No Image with name exist {mapped_img[0]}")
                        print(f"Augmentation completed for class {self.tag_list[list_index]}")
                        self.custom_logger.info(f"Augmentation completed for class {self.tag_list[list_index]}")
                    print('Successfully Performed Generic Standard Augmentations on parts of classes only to have '
                          'reduced data ')
                    self.custom_logger.info('Successfully Performed Generic Standard Augmentations on parts of '
                                            'classes only to have reduced data')
            else:
                print('NO Generic Standard Augmentations performed')
                self.custom_logger.info('NO Generic Standard Augmentations performed')
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

    def perform_pipeline_operation(self):
        if self.custom_pipeline_yolo_version == "YoloV7":
            # Yolo v-7
            print('Started Creating Repo for local Training of Yolo v-7 model')
            os.mkdir('yolov7-main/data/train')
            os.mkdir('yolov7-main/data/train/images')
            os.mkdir('yolov7-main/data/train/labels')
            os.mkdir('yolov7-main/data/val')
            os.mkdir('yolov7-main/data/val/images')
            os.mkdir('yolov7-main/data/val/labels')
            os.mkdir('yolov7-main/data/val/json')
            print('Completed Creating Repo for local Training of Yolo v-7 model')
            print('Started Moving Data for local Training of Yolo v-7 model')
            for img_file in os.listdir("training_data/Images/items"):
                shutil.copy(f'training_data/Images/items/{img_file}', f'yolov7-main/data/train/images/{img_file}')

            for txt_file in os.listdir("training_data/labels/yolo"):
                shutil.copy(f'training_data/labels/yolo/{txt_file}', f'yolov7-main/data/train/labels/{txt_file}')

            for file in os.listdir("groundtruth_testing_data/data"):
                try:
                    if file.endswith(('.png', '.jpg', '.jpeg')):
                        shutil.copy(f'groundtruth_testing_data/data/{file}', f'yolov7-main/data/val/images/{file}')
                    elif file.endswith('.txt'):
                        shutil.copy(f'groundtruth_testing_data/data/{file}', f'yolov7-main/data/val/labels/{file}')
                    else:
                        shutil.copy(f'groundtruth_testing_data/data/{file}', f'yolov7-main/data/val/json/{file}')
                except Exception as e:
                    print(e)
            print('Completed Moving Data for local Training of Yolo v-7 model')
            os.system(
                f'python3 yolov7-main/train.py --workers {self.workers} --device {self.device} --batch-size {self.batch_size} --data {self.data_yaml} --img 640 640 --cfg {self.cfg_yaml} --weights {self.weights} --name {self.model_name} --hyp {self.hyper_parameter_file}')
        elif self.custom_pipeline_yolo_version == "YoloV8":
            # # YOLOv8
            # print('Started Creating Repo for local Training of YOLOv8 model')
            
            # # Check if training data exists
            # if not os.path.exists("training_data/Images/items") or not os.listdir("training_data/Images/items"):
            #     print("ERROR: No training data found in training_data/Images/items/")
            #     print("Please run data preparation first:")
            #     print("1. Check if raw data exists in 'data/' directory")
            #     print("2. Run data preprocessing using raw_training_data.py or similar")
            #     print("3. Ensure images are organized in training_data/Images/items/")
            #     print("4. Ensure labels are organized in training_data/labels/yolo/")
            #     return
            
            # if not os.path.exists("training_data/labels/yolo") or not os.listdir("training_data/labels/yolo"):
            #     print("ERROR: No training labels found in training_data/labels/yolo/")
            #     print("Please ensure YOLO format labels are available before training.")
            #     return
            
            # train_image_count = len(os.listdir("training_data/Images/items"))
            # train_label_count = len(os.listdir("training_data/labels/yolo"))
            # print(f"Found {train_image_count} training images and {train_label_count} labels")
            
            # os.makedirs('yolov8-main/data/train/images', exist_ok=True)
            # os.makedirs('yolov8-main/data/train/labels', exist_ok=True)
            # os.makedirs('yolov8-main/data/val/images', exist_ok=True)
            # os.makedirs('yolov8-main/data/val/labels', exist_ok=True)
            # os.makedirs('yolov8-main/data/val/json', exist_ok=True)
            # print('Completed Creating Repo for local Training of YOLOv8 model')
            # print('Started Moving Data for local Training of YOLOv8 model')
            
            # # Copy training data
            # for img_file in os.listdir("training_data/Images/items"):
            #     shutil.copy(f'training_data/Images/items/{img_file}', f'yolov8-main/data/train/images/{img_file}')

            # for txt_file in os.listdir("training_data/labels/yolo"):
            #     shutil.copy(f'training_data/labels/yolo/{txt_file}', f'yolov8-main/data/train/labels/{txt_file}')

            # Copy validation data (use copy instead of move to preserve for GT testing)
            # gt_data_source = "groundtruth_testing_data/data"
            # if os.path.exists(gt_data_source) and os.listdir(gt_data_source):
            #     print(f"Using existing ground truth data from {gt_data_source}")
            #     for file in os.listdir(gt_data_source):
            #         try:
            #             if file.endswith(('.png', '.jpg', '.jpeg')):
            #                 shutil.copy(f'{gt_data_source}/{file}', f'yolov8-main/data/val/images/{file}')
            #             elif file.endswith('.txt'):
            #                 shutil.copy(f'{gt_data_source}/{file}', f'yolov8-main/data/val/labels/{file}')
            #             elif file.endswith('.json'):
            #                 shutil.copy(f'{gt_data_source}/{file}', f'yolov8-main/data/val/json/{file}')
            #         except Exception as e:
            #             print(f"Error copying validation file {file}: {e}")
            # else:
            #     print("No ground truth data found. Creating validation split from training data...")
            #     # Create validation split from training data
            #     self._create_validation_split_for_yolov8()
                
            #     # Also populate groundtruth_testing_data for later GT testing
            #     self._populate_gt_testing_data_from_validation()
            # print('Completed Moving Data for local Training of YOLOv8 model')
            print("yolov8 training _started")
            
            # YOLOv8 training using Ultralytics
            try:
                from ultralytics import YOLO
                from ultralytics import settings
                import mlflow
                import uuid
                
                # Set MLflow tracking URI to point to the correct mlruns folder (one level up)
                mlflow.set_tracking_uri("mlruns")
                os.environ['MLFLOW_TRACKING_URI'] = 'mlruns'
                print("MLflow tracking URI set to: mlruns (one folder back)")
                
                # ENABLE YOLOv8's built-in MLflow integration with correct path
                settings.update({"mlflow": True})
                print("YOLOv8 built-in MLflow integration ENABLED - will save to ../mlruns")
                
                # Generate custom iteration ID similar to YOLOv4
                custom_iteration_id = str(uuid.uuid1())
                
                # Initialize model
                model_size = self.custom_config_dict.get("model_size", "yolov8n.pt")
                model = YOLO(model_size)
                
                # Verify the data YAML has the correct number of classes
                print(f"Validating class configuration for YOLOv8...")
                try:
                    import yaml
                    with open(yolov8_data_yaml, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                    
                    if 'nc' in yaml_data and yaml_data['nc'] != 3:
                        print(f"Warning: YAML nc={yaml_data['nc']} but expected 5 classes")
                        print(f"This might cause background class issues in YOLOv8")
                    
                    if 'names' in yaml_data:
                        class_count = len(yaml_data['names']) if isinstance(yaml_data['names'], list) else len(yaml_data['names'])
                        if class_count != 5:
                            print(f"Warning: Found {class_count} class names but expected 5")
                        else:
                            print(f"✅ Correct 5 classes found: {yaml_data['names']}")
                except Exception as e:
                    print(f"Warning: Could not validate YAML config: {e}")
                
                # Get training parameters
                epochs = self.custom_config_dict.get("epochs", 100)
                imgsz = self.custom_config_dict.get("imgsz", 640)
                patience = self.custom_config_dict.get("patience", 50)
                save_period = self.custom_config_dict.get("save_period", 10)
                # save_period = self.custom_config_dict.get("save_period", 10)
                
                # Use the data_yaml from config, but validate it's a YAML file for YOLOv8
                yolov8_data_yaml = self.data_yaml
                if not yolov8_data_yaml.endswith('.yaml') and not yolov8_data_yaml.endswith('.yml'):
                    print(f"Warning: YOLOv8 requires YAML format, but got: {yolov8_data_yaml}")
                    print("YOLOv8 data file should end with .yaml or .yml extension")
                    raise ValueError(f"Invalid data file format for YOLOv8: {yolov8_data_yaml}")
                
                if not os.path.exists(yolov8_data_yaml):
                    raise FileNotFoundError(f"YOLOv8 data configuration file not found: {yolov8_data_yaml}")
                
                print(f"Using YOLOv8 data config: {yolov8_data_yaml}")
                print(f"Starting YOLOv8 training with {epochs} epochs...")
                print(f"Custom Iteration ID: {custom_iteration_id}")
                
                # # Train the model WITHOUT built-in MLflow logging (disabled to use custom logging)
                results = model.train(
                    data=yolov8_data_yaml,
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=self.batch_size,
                    device=self.device,
                    workers=self.workers,
                    project='./object_detection_yolov8_training',
                    name=self.model_name,
                    patience=patience,
                    save_period=save_period,
                    exist_ok=True,
                    verbose=True,
                    auto_augment=self.custom_config_dict.get("auto_augment", None),
                    cfg=self.custom_config_dict.get("hyperparameters_yaml", "./framework_config/hyperparameters.yaml")
                    # mosaic=self.custom_config_dict.get("mosaic", None)
                )
                # After training, get metrics
                metrics = results.results_dict
                precision = metrics.get('metrics/precision', 0)
                recall = metrics.get('metrics/recall', 0)

                # Calculate F1 score
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1_score:.4f}")
                print(f"mAP50: {metrics.get('metrics/mAP50', 0):.4f}")
                print(f"mAP50-95: {metrics.get('metrics/mAP50-95', 0):.4f}")
                
                print("YOLOv8 training completed successfully!")
                
                # Copy trained models to model_registry
                source_folder = f'./object_detection_yolov8_training/{self.model_name}/weights/'
                destination_folder = './model_registry/'
                
                # Ensure model_registry directory exists
                if not os.path.exists(destination_folder):
                    os.makedirs(destination_folder)
                    
                model_list = []
                if os.path.exists(source_folder):
                    for file_name in os.listdir(source_folder):
                        if file_name.endswith('.pt'):
                            source = os.path.join(source_folder, file_name)
                            destination = os.path.join(destination_folder, f"yolov8_{custom_iteration_id}_{file_name}")
                            shutil.copy(source, destination)
                            model_list.append(f"yolov8_{custom_iteration_id}_{file_name}")
                            print('Copied YOLOv8 Model generated successfully', f"yolov8_{custom_iteration_id}_{file_name}")
                
                print(f"YOLOv8 training completed with {len(model_list)} models generated")
                print("YOLOv8 MLflow logging handled automatically by Ultralytics - check ../mlruns folder and MLflow UI")
            except Exception as e:
                print(e)       
        #         # YOLOv8 Ground Truth Testing (following YOLOv4 approach exactly)
        #         print("Available trained YOLOv8 models for GT testing:")
        #         ###addign for validation only if you are going for training comment this line.
        #         model_list= os.listdir('./model_registry/')
        #         for idx, model in enumerate(model_list):
        #             print(f"{idx + 1}. {model}")
                
        #         model_value = input("Please Enter Model from above list only on which you want to perform "
        #                             "Groundtruth testing on: ")
                
        #         # Create GT testing directories exactly like YOLOv4 does
        #         os.mkdir('./groundtruth_testing_data/data/originalimages')
        #         os.mkdir('./groundtruth_testing_data/data/original Annotation')
        #         os.mkdir('./groundtruth_testing_data/data/original Json')
        #         os.mkdir('./groundtruth_testing_data/data/predicted')
                
        #         # Copy files from rawtraining-data structure to groundtruth_testing_data
        #         print("Copying files from rawtraining-data to groundtruth_testing_data...")
        #         rawtraining_base = 'rawtraining-data'  # FIXED: Correct path structure
        #         gt_data_base = './groundtruth_testing_data/data'
                
        #         if os.path.exists(rawtraining_base):
        #             total_copied = 0
        #             # Copy classes.names file
        #             classes_file = os.path.join(rawtraining_base, 'classes.names')
        #             if os.path.exists(classes_file):
        #                 shutil.copy(classes_file, os.path.join(gt_data_base, 'classes.names'))
        #                 print(f"Copied classes.names file")
        #             else:
        #                 print(f"Warning: classes.names not found at {classes_file}")
                    
        #             # Process each valve directory
        #             for valve_dir in ['valve-Blind', 'valve-check', 'valve-gate', 'valve-globe', 'valve-Indicator']:
        #                 valve_path = os.path.join(rawtraining_base, valve_dir)
        #                 if os.path.exists(valve_path):
        #                     print(f"Processing {valve_dir}...")
        #                     for file in os.listdir(valve_path):
        #                         file_path = os.path.join(valve_path, file)
        #                         if os.path.isfile(file_path):
        #                             # Determine valve type for filename prefix
        #                             valve_type_map = {
        #                                 'valve-Blind': 'spectacle_Blind',
        #                                 'valve-check': 'check_valve',
        #                                 'valve-gate': 'gate_valve',
        #                                 'valve-globe': 'globe_valve',
        #                                 'valve-Indicator': 'Pressure_Indicator'
        #                             }
        #                             valve_type = valve_type_map[valve_dir]
                                    
        #                             # Create new filename with valve type prefix
        #                             new_filename = f"{valve_type}_{file}"
        #                             dest_path = os.path.join(gt_data_base, new_filename)
                                    
        #                             shutil.copy(file_path, dest_path)
        #                             total_copied += 1
                                    
        #                             # Create corresponding JSON metadata file for images
        #                             if file.endswith(('.png', '.jpg', '.jpeg')):
        #                                 json_filename = f"{valve_type}_{os.path.splitext(file)[0]}.json"
        #                                 json_path = os.path.join(gt_data_base, json_filename)
                                        
        #                                 json_data = {
        #                                     "image_name": os.path.splitext(new_filename)[0],
        #                                     "user": {
        #                                         "valve": valve_type,
        #                                         "rawtraining": "rawtraining-data"
        #                                     }
        #                                 }
                                        
        #                                 with open(json_path, 'w') as f:
        #                                     json.dump(json_data, f, indent=2)
        #                                 total_copied += 1
                    
        #             print(f"Copied {total_copied} files from rawtraining-data to groundtruth_testing_data")
        #         else:
        #             print("Warning: rawtraining-data directory not found!")
                
        #         # Organize ground truth testing data exactly like YOLOv4
        #         source = './groundtruth_testing_data/data/'
        #         mydict = {
        #             './groundtruth_testing_data/data/originalimages': ['jpg','png','jpeg'],
        #             './groundtruth_testing_data/data/original Annotation': ['txt'],
        #             './groundtruth_testing_data/data/original Json': ['json']
        #         }
        #         for destination, extensions in mydict.items():
        #             for ext in extensions:
        #                 for file in glob.glob(source + '*.' + ext):
        #                     print(file)
        #                     shutil.move(file, destination)
                
        #         # Validate and fix ground truth annotation format
        #         self._validate_and_fix_gt_annotations('./groundtruth_testing_data/data/original Annotation')
                
        #         # Run YOLOv8 inference exactly like YOLOv4 approach
        #         cmd = f'yolo predict model=./model_registry/{model_value} source=./groundtruth_testing_data/data/originalimages/ save_txt=True conf=0.4 project=./groundtruth_testing_data/data name=predicted exist_ok=True'
        #         os.system(cmd)
                
        #         # Convert YOLO format predictions to absolute coordinate format
        #         self._convert_yolo_predictions_to_absolute('./groundtruth_testing_data/data/predicted', 
        #                                                    './groundtruth_testing_data/data/originalimages')
                
        #         # Rename originalimages to "original images" exactly like YOLOv4
        #         os.rename('./groundtruth_testing_data/data/originalimages', './groundtruth_testing_data/data/original images')
        #           # Copy names file exactly like YOLOv4
        #         shutil.copyfile(f'./training_data/labels/{self.dataset_name}.names', f'./groundtruth_testing_data/data/{self.dataset_name}.names')
                
        #         # Generate comprehensive Ground truth Summary report for YOLOv8
        #         print(f"Generating comprehensive GT reports for YOLOv8")
        #         try:
        #             import pandas as pd
        #             import yaml
        #             import json
        #             import mlflow
        #             import datetime
        #             import traceback
        #             from metrics_calculation import calculate_metrics
        #             from class_mapping import get_class_mapping, get_class_name, normalize_valve_type
        #             # Import the correct export function (not the split_sheet version)
        #             from generate_overall_summary import export_to_summary_excel
        #             from generate_detailed_gt import generate_detailed_report
                    
        #             # Validate required directories exist
        #             required_dirs = [
        #                 './groundtruth_testing_data/data/original Annotation/',
        #                 './groundtruth_testing_data/data/predicted/',
        #                 './groundtruth_testing_data/data/original images/'
        #             ]
                    
        #             for dir_path in required_dirs:
        #                 if not os.path.exists(dir_path):
        #                     print(f"Error: Required directory not found: {dir_path}")
        #                     raise FileNotFoundError(f"Missing directory: {dir_path}")
        #                 if not os.listdir(dir_path):
        #                     print(f"Warning: Directory is empty: {dir_path}")
                    
        #             # Load groundtruth and prediction data from YOLO format files with robust error handling
        #             data_df_path = './groundtruth_testing_data/data/data_df.json' 
        #             if os.path.exists(data_df_path):
        #                 try:
        #                     data_df = pd.read_json(data_df_path)
        #                     print(f"Loaded metadata from {data_df_path}: {len(data_df)} entries")
        #                     # Validate required columns exist
        #                     required_columns = ['image', 'rawtraining', 'valve']
        #                     for col in required_columns:
        #                         if col not in data_df.columns:
        #                             print(f"Warning: Missing column '{col}' in metadata. Adding values.")
        #                             data_df[col] = 'unknown'
        #                     # Ensure DataFrame has proper data types and reset index
        #                     data_df = data_df.astype(str).reset_index(drop=True)
        #                     print(f"DataFrame structure validated. Shape: {data_df.shape}")
        #                 except Exception as e:
        #                     print(f"Error loading metadata file: {e}. Creating empty structure.")
        #                     data_df = pd.DataFrame({'image': [], 'rawtraining': [], 'valve': []})                    
        #             else:
        #                 print(f"Warning: Metadata file not found at {data_df_path}. Creating structure from JSON files.")
        #                 # Create metadata structure from individual JSON files in "original Json" folder
        #                 json_metadata_dir = './groundtruth_testing_data/data/original Json/'
        #                 metadata_records = []
                        
        #                 if os.path.exists(json_metadata_dir):
        #                     json_files = [f for f in os.listdir(json_metadata_dir) if f.endswith('.json')]
        #                     print(f"Found {len(json_files)} JSON metadata files")
                            
        #                     for json_file in json_files:
        #                         try:
        #                             json_path = os.path.join(json_metadata_dir, json_file)
        #                             with open(json_path, 'r') as f:
        #                                 json_data = json.load(f)
                                    
        #                             if 'user' in json_data:
        #                                 user_data = json_data['user']
        #                                 # Extract image name from JSON filename
        #                                 image_name = os.path.splitext(json_file)[0]
                                        
        #                                 # Create metadata record with proper fallbacks
        #                                 metadata_record = {
        #                                     'image': image_name,
        #                                     'rawtraining': user_data.get('rawtraining', 'unknown'),
        #                                     'valve': user_data.get('valve', user_data.get('valve', 'unknown')),  # Handle case variations
        #                                     #'lighting': user_data.get('lighting', 'unknown'),
        #                                     #'distance': user_data.get('distance', user_data.get('Distance', 'unknown'))  # Handle case variations
        #                                 }
        #                                 metadata_records.append(metadata_record)
        #                                 print(f"Loaded metadata for {image_name}: {metadata_record}")
        #                             else:
        #                                 print(f"Warning: No 'user' key in {json_file}")
        #                         except Exception as e:
        #                             print(f"Warning: Error reading {json_file}: {e}")
        #                             continue
                        
        #                 if metadata_records:
        #                     data_df = pd.DataFrame(metadata_records)
        #                     print(f"Successfully created metadata DataFrame from JSON files: {len(data_df)} entries")
        #                 else:
        #                     print("No valid JSON metadata found. Using empty structure.")
        #                     data_df = pd.DataFrame({'image': [], 'rawtraining': [], 'valve': []})
                    
        #             # Load class names from .names file for proper labeling
        #             class_names = []
        #             # Try multiple possible locations for classes.names
        #             possible_names_files = [
        #                 f'./groundtruth_testing_data/data/{self.dataset_name}.names',
        #                 './groundtruth_testing_data/data/classes.names',
        #                 'rawtraining-data/classes.names'
        #             ]
                    
        #             for names_file_path in possible_names_files:
        #                 if os.path.exists(names_file_path):
        #                     try:
        #                         with open(names_file_path, 'r') as f:
        #                             class_names = [line.strip() for line in f.readlines() if line.strip()]
        #                         print(f"Loaded {len(class_names)} class names from {names_file_path}: {class_names}")
        #                         break
        #                     except Exception as e:
        #                         print(f"Error loading class names from {names_file_path}: {e}")
        #                         continue
                    
        #             if not class_names:
        #                 # Fallback to hardcoded class names based on rawtraining-data structure
        #                 class_names = ['gate_valve', 'globe_valve', 'check_valve', 'spectacle_Blind', 'Pressure_Indicator']
        #                 print(f"Warning: No classes.names file found. Using hardcoded classes: {class_names}")
                    
        #             def get_class_name(class_id):
        #                 """Get proper class name from class ID"""
        #                 try:
        #                     if class_names and 0 <= class_id < len(class_names):
        #                         return class_names[class_id]
        #                     else:
        #                         return f'class_{class_id}'
        #                 except:
        #                     return f'class_{class_id}'
                    
        #             # Parse ground truth data with proper class names
        #             groundtruth_details = []
        #             gt_annotation_path = './groundtruth_testing_data/data/original Annotation/'
        #             gt_files = [f for f in os.listdir(gt_annotation_path) if f.endswith('.txt')]
        #             print(f"Found {len(gt_files)} ground truth annotation files")
                    
        #             for annotation_file in gt_files:
        #                 image_name = annotation_file.replace('.txt', '')
        #                 file_path = os.path.join(gt_annotation_path, annotation_file)
        #                 with open(file_path, 'r') as f:
        #                     lines = f.readlines()
        #                     for line_num, line in enumerate(lines, 1):
        #                         line = line.strip()
        #                         if not line:  # Skip empty lines
        #                             continue
        #                         parts = line.split()
        #                         if len(parts) >= 5:
        #                             try:
        #                                 class_id, x_center, y_center, width, height = map(float, parts[:5])
        #                                 class_id = int(class_id)
                                        
        #                                 # Validate class ID is within expected range (0-4 for 5 valve classes)
        #                                 if not (0 <= class_id <= 4):
        #                                     print(f"Warning: Invalid class ID {class_id} in {annotation_file} line {line_num} - skipping (prevents background class)")
        #                                     continue
                                        
        #                                 class_name = get_class_name(class_id)
        #                                 if class_name is None:
        #                                     print(f"Warning: Could not map class ID {class_id} in {annotation_file} line {line_num} - skipping")
        #                                     continue
                                        
        #                                 groundtruth_details.append({
        #                                     'image_name': image_name,
        #                                     'class_name': class_name,
        #                                     'bounding_box': [x_center, y_center, width, height]
        #                                 })
        #                             except ValueError as e:
        #                                 print(f"Error parsing GT file {annotation_file}, line {line_num}: {line} - {e}")
        #                                 continue
        #                         else:
        #                             print(f"Warning: Invalid GT annotation format in {annotation_file}, line {line_num}: {line}")
                    
        #             print(f"Parsed {len(groundtruth_details)} ground truth annotations")
                    
        #             # Parse prediction data with proper class names
        #             prediction_details = []
        #             pred_annotation_path = './groundtruth_testing_data/data/predicted/'
        #             pred_files = [f for f in os.listdir(pred_annotation_path) if f.endswith('.txt')]
                  
                    
        #             # Helper function to convert absolute coordinates back to YOLO format for metrics
        #             def convert_absolute_to_yolo(x1, y1, x2, y2, img_width, img_height):
        #                 x_center = (x1 + x2) / 2 / img_width
        #                 y_center = (y1 + y2) / 2 / img_height
        #                 width = (x2 - x1) / img_width
        #                 height = (y2 - y1) / img_height
        #                 return [x_center, y_center, width, height]
                    
        #             for prediction_file in pred_files:
        #                 image_name = prediction_file.replace('.txt', '')
        #                 file_path = os.path.join(pred_annotation_path, prediction_file)
                        
        #                 # Get image dimensions for coordinate conversion
        #                 image_path = None
        #                 for ext in ['jpg', 'jpeg', 'png']:
        #                     potential_path = f'./groundtruth_testing_data/data/original images/{image_name}.{ext}'
        #                     if os.path.exists(potential_path):
        #                         image_path = potential_path
        #                         break
                        
        #                 if image_path and os.path.exists(image_path):
        #                     import cv2
        #                     img = cv2.imread(image_path)
        #                     if img is not None:
        #                         img_height, img_width = img.shape[:2]
        #                     else:
        #                         print(f"Warning: Could not load image {image_path}")
        #                         continue
        #                 else:
        #                     print(f"Warning: Could not find image for {image_name}")
        #                     continue
                        
        #                 with open(file_path, 'r') as f:
        #                     lines = f.readlines()
        #                     for line_num, line in enumerate(lines, 1):
        #                         line = line.strip()
        #                         if not line:  # Skip empty lines
        #                             continue
        #                         parts = line.split()
                                
        #                         # Handle both YOLO format (6 parts) and absolute format (6+ parts)
        #                         if len(parts) >= 6:
        #                             try:
        #                                 class_id = int(float(parts[0]))
        #                                 confidence = float(parts[1])
                                        
        #                                 # Check if coordinates are in absolute format (converted)
        #                                 if len(parts) == 6 and float(parts[2]) > 1 or float(parts[3]) > 1:
        #                                     # Absolute coordinate format: class_id confidence x1 y1 x2 y2
        #                                     x1, y1, x2, y2 = map(float, parts[2:6])
        #                                     # Convert to YOLO format for consistency
        #                                     yolo_bbox = convert_absolute_to_yolo(x1, y1, x2, y2, img_width, img_height)
        #                                 else:
        #                                     # YOLO format: class_id confidence x_center y_center width height
        #                                     yolo_bbox = [float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                                        
        #                                 prediction_details.append({
        #                                     'image_name': image_name,
        #                                     'class_name': get_class_name(class_id),
        #                                     'bounding_box': yolo_bbox,
        #                                     'confidence_score': confidence,
        #                                     'category': None  # Will be filled by add_category_to_data
        #                                 })
        #                                 print(f"DEBUG: Added prediction for {image_name}, class {get_class_name(class_id)}, conf {confidence:.3f}")
        #                             except ValueError as e:
        #                                 print(f"Error parsing prediction file {prediction_file}, line {line_num}: {line} - {e}")
        #                                 continue
        #                         elif len(parts) >= 5:
        #                             try:
        #                                 # Handle YOLO format without confidence: class_id x_center y_center width height
        #                                 class_id = int(float(parts[0]))
        #                                 x_center, y_center, width, height = map(float, parts[1:5])
        #                                 confidence = 0.5  # Standard confidence for YOLO format
                                        
        #                                 prediction_details.append({
        #                                     'image_name': image_name,
        #                                     'class_name': get_class_name(class_id),
        #                                     'bounding_box': [x_center, y_center, width, height],
        #                                     'confidence_score': confidence,
        #                                     'category': None  # Will be filled by add_category_to_data
        #                                 })
        #                                 print(f"DEBUG: Added YOLO prediction for {image_name}, class {get_class_name(class_id)}")
        #                             except ValueError as e:
        #                                 print(f"Error parsing YOLO prediction file {prediction_file}, line {line_num}: {line} - {e}")
        #                                 continue
        #                         else:
        #                             print(f"Warning: Invalid prediction format in {prediction_file}, line {line_num}: {line} (expected at least 5 parts, got {len(parts)})")
                    
        #             print(f"Parsed {len(prediction_details)} prediction annotations")
                    
        #             if not groundtruth_details:
        #                 print("Error: No valid ground truth annotations found!")
        #                 print("Please check that:")
        #                 print("1. Ground truth annotation files exist in ./groundtruth_testing_data/data/original Annotation/")
        #                 print("2. Files are in YOLO format: class_id x_center y_center width height")
        #                 print("3. Files contain valid numerical data")
        #                 raise ValueError("No ground truth data available for processing")
                    
        #             if not prediction_details:
        #                 print("Warning: No valid prediction annotations found!")
        #                 print("This may indicate:")
        #                 print("1. Model made no predictions above confidence threshold")
        #                 print("2. Prediction files are empty or in wrong format")
        #                 print("3. YOLOv8 inference failed")
        #                 print("Proceeding with empty predictions (all ground truth will be missed detections)")
                    
        #             # Generate comprehensive metrics and Excel reports with enhanced error handling
        #             iou_threshold = 0.5
        #             print(f"Starting metrics calculation with IOU threshold: {iou_threshold}")
                    
        #             # Create a safer data_df structure to prevent pandas indexing errors
        #             try:
        #                 # Use the existing data_df properly
        #                 safe_data_df = data_df.copy().reset_index(drop=True)
        #                 print(f"DataFrame created. Shape: {safe_data_df.shape}")
                        
        #             except Exception as setup_error:
        #                 print(f"Error setting up data structure: {setup_error}")
        #                 return
                    
        #             # Call calculate_metrics to get proper results
        #             try:
        #                 report_folder = calculate_metrics(groundtruth_details, prediction_details, safe_data_df, iou_threshold)
        #                 print(f"Metrics calculation completed. Reports saved to: {report_folder}")
                        
        #                 # The calculate_metrics function already generates all Excel reports including overall_summary.xlsx
        #                 # Check if the summary was created successfully
        #                 summary_file = os.path.join(report_folder, "overall_summary.xlsx")
        #                 if os.path.exists(summary_file):
        #                     print(f"Overall summary Excel report generated successfully at: {summary_file}")
        #                 else:
        #                     print("Warning: Overall summary Excel was not generated by calculate_metrics")
                            
        #             except Exception as calc_error:
        #                 print(f"Error during metrics calculation: {calc_error}")
        #                 print("Creating report folder...")
        #                 import datetime
        #                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        #                 report_folder = f"groundtruth_testing_data/data/{timestamp}_results"
        #                 os.makedirs(report_folder, exist_ok=True)
        #                 print(f"Report folder created: {report_folder}")
                    
        #             # Generate detailed report with images and annotations - with JSON file fix
        #             if os.path.exists('./groundtruth_testing_data/data/original images/'):
        #                 print("Generating detailed report with images...")
        #                 # Use actual class names from the data
        #                 unique_classes = list(set(gt['class_name'] for gt in groundtruth_details if gt.get('class_name')))
        #                 tag_list = unique_classes if unique_classes else ['gate_valve', 'globe_valve', 'check_valve', 'spectacle_Blind', 'Pressure_Indicator']
                        
        #                 try:
        #                     # Create dummy JSON files if they don't exist to prevent FileNotFoundError
        #                     json_dir = './groundtruth_testing_data/data/original Json/'
        #                     if not os.path.exists(json_dir):
        #                         os.makedirs(json_dir, exist_ok=True)
                            
        #                     # Create minimal JSON files for images that don't have them
        #                     image_files = [f.replace('.txt', '') for f in os.listdir('./groundtruth_testing_data/data/original Annotation/') if f.endswith('.txt')]                            
        #                     for image_name in image_files:
        #                         json_file_path = os.path.join(json_dir, f"{image_name}.json")
        #                         if not os.path.exists(json_file_path):
        #                             # Create minimal JSON with default metadata matching expected structure
        #                             default_json = {
        #                                 "image_name": image_name,
        #                                 "user": {
        #                                     "valve": "unknown",
        #                                     #"lighting": "unknown",
        #                                     #"distance": "unknown",
        #                                     "rawtraining": "unknown"
        #                                 }
        #                             }
        #                             with open(json_file_path, 'w') as f:
        #                                 json.dump(default_json, f, indent=2)
        #                     generate_detailed_report(
        #                         report_path=report_folder,
        #                         images_path='./groundtruth_testing_data/data/original images/',
        #                         annotation_path='./groundtruth_testing_data/data/original Annotation/',
        #                         predicted_annotation_path='./groundtruth_testing_data/data/predicted/',
        #                         metadata_path='./groundtruth_testing_data/data/original Json/',
        #                         tag_list=tag_list,
        #                         inference_image_path=os.path.join(report_folder, 'inference_images')
        #                     )
        #                     print("Detailed report generation completed")                        
        #                 except Exception as detail_error:
        #                         print(f"Warning: Detailed report generation failed: {detail_error}")
        #                         print("Continuing with other reports...")
                    
        #             # LOG GT METRICS TO MLFLOW
        #             try:
        #                 print("Logging GT metrics to MLflow...")
                        
        #                 # Set up MLflow experiment for GT testing
        #                 experiment_name = f"Object Detection YOLOv8_GT_Testing_{self.model_name}" if self.model_name else "YOLOv8_GT_Testing"
        #                 mlflow.set_experiment(experiment_name)
                        
        #                 # Start MLflow run for GT metrics
        #                 with mlflow.start_run(run_name=f"YOLOv8_GT_Testing_{custom_iteration_id}"):
        #                     # Log GT testing parameters
        #                     mlflow.log_param("model_used", model_value)
        #                     mlflow.log_param("iou_threshold", iou_threshold)
        #                     mlflow.log_param("total_gt_annotations", len(groundtruth_details))
        #                     mlflow.log_param("total_predictions", len(prediction_details))
        #                     mlflow.log_param("gt_testing_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            
        #                     # Log GT metrics
        #                     if overall_summary_data['overall_metrics']:
        #                         metrics = overall_summary_data['overall_metrics']
        #                         mlflow.log_metric("gt_precision", metrics['precision'])
        #                         mlflow.log_metric("gt_recall", metrics['recall'])
        #                         mlflow.log_metric("gt_f1_score", metrics['f1'])
        #                         mlflow.log_metric("gt_true_positives", metrics['tp'])
        #                         mlflow.log_metric("gt_false_positives", metrics['fp'])
        #                         mlflow.log_metric("gt_false_negatives", metrics['fn'])
        #                         mlflow.log_metric("gt_data_size", metrics['data size'])
                            
        #                     # Log class information if available
        #                     if groundtruth_details:
        #                         unique_classes = list(set(gt['class_name'] for gt in groundtruth_details if gt.get('class_name')))
        #                         for class_name in unique_classes:
        #                             class_gt_count = len([gt for gt in groundtruth_details if gt['class_name'] == class_name])
        #                             class_pred_count = len([pred for pred in prediction_details if pred['class_name'] == class_name])
        #                             mlflow.log_metric(f"gt_class_{class_name}_ground_truth_count", class_gt_count)
        #                             mlflow.log_metric(f"gt_class_{class_name}_prediction_count", class_pred_count)
                            
        #                     # Log artifacts - Excel reports
        #                     if os.path.exists(report_folder):
        #                         for file in os.listdir(report_folder):
        #                             if file.endswith('.xlsx'):
        #                                 mlflow.log_artifact(os.path.join(report_folder, file), "gt_reports")
        #                                 print(f"GT metrics logged to MLflow successfully!")
                    
        #             except Exception as mlflow_error:
        #                 print(f"Warning: MLflow GT logging failed: {mlflow_error}")
        #                 print(f"Error type: {type(mlflow_error).__name__}")
        #                 print("GT reports generated but not logged to MLflow")
        #                 # Print more detailed error information
        #                 print("Detailed error traceback:")
        #                 traceback.print_exc()
                    
        #             print(f"Generated comprehensive GT reports for YOLOv8 in: {report_folder}")
                    
        #             # Copy summary report to gt_reports for compatibility
        #             if os.path.exists(report_folder):
        #                 gt_reports_dir = './gt_reports'
        #                 os.makedirs(gt_reports_dir, exist_ok=True)
        #                 try:
        #                     shutil.copytree(report_folder, os.path.join(gt_reports_dir, os.path.basename(report_folder)), dirs_exist_ok=True)
        #                     print(f"Reports copied to {gt_reports_dir} for compatibility")
        #                 except Exception as copy_error:
        #                     print(f"Warning: Could not copy reports to gt_reports: {copy_error}")
                    
        #             # Generate summary charts and upload to MLflow after all Excel reports are completed
        #             print("Generating summary charts and uploading to MLflow...")
        #             try:
        #                 from generate_summary_representation import generate_summary_graph
                        
        #                 # Create config structure required by summary_report.py
        #                 val_config = {
        #                     'report_generator': {
        #                         'summary_header': ['valve']  # Default headers for summary report
        #                     }
        #                 }
                        
        #                 # Generate summary graphs and charts
        #                 generate_summary_graph(report_folder, val_config)
        #                 print(f"Summary charts generated successfully in {report_folder}/Merged_Comparison_Charts")
                        
        #                 # Upload charts to MLflow if available
        #                 try:
        #                     chart_folder = os.path.join(report_folder, "Merged_Comparison_Charts")
        #                     if os.path.exists(chart_folder):
        #                         # Log charts to current MLflow run if one is active
        #                         import mlflow
        #                         if mlflow.active_run():
        #                             for chart_file in os.listdir(chart_folder):
        #                                 if chart_file.endswith('.png'):
        #                                     chart_path = os.path.join(chart_folder, chart_file)
        #                                     mlflow.log_artifact(chart_path, "summary_charts")
        #                                     print(f"Uploaded chart {chart_file} to MLflow")
        #                         else:
        #                             # Start a new MLflow run for chart logging
        #                             with mlflow.start_run(run_name=f"YOLOv8_Summary_Charts_{custom_iteration_id}"):
        #                                 for chart_file in os.listdir(chart_folder):
        #                                     if chart_file.endswith('.png'):
        #                                         chart_path = os.path.join(chart_folder, chart_file)
        #                                         mlflow.log_artifact(chart_path, "summary_charts")
        #                                         print(f"Uploaded chart {chart_file} to MLflow")
                                        
        #                                 # Log chart generation parameters
        #                                 mlflow.log_param("chart_generation_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #                                 mlflow.log_param("yolo_version", "YOLOv8")
        #                                 mlflow.log_param("model_used", model_value)
        #                                 mlflow.log_metric("total_charts_generated", len([f for f in os.listdir(chart_folder) if f.endswith('.png')]))
                                        
        #                         print("All summary charts uploaded to MLflow successfully!")
        #                     else:
        #                         print(f"Chart folder not found at {chart_folder}")
        #                 except Exception as mlflow_chart_error:
        #                     print(f"Warning: Failed to upload charts to MLflow: {mlflow_chart_error}")
                            
        #             except Exception as chart_error:
        #                 print(f"Warning: Failed to generate summary charts: {chart_error}")
        #                 print("Detailed error:")
        #                 traceback.print_exc()
                        
        #         except Exception as gt_error:
        #                 print(f"Error during YOLOv8 GT report generation: {gt_error}")
        #                 print("Check that required files and directories exist")
        #                 traceback.print_exc()
                
        #         print(f"YOLOv8 ground truth testing and reporting completed.")
                
        #     except ImportError:
        #         print("Ultralytics library not found. Please install it using: pip install ultralytics")
        #         print("Using command line training...")
        #         # Command line training with config-based data file
        #         training_command = f'yolo detect train data={yolov8_data_yaml} model={self.custom_config_dict.get("model_size", "yolov8n.pt")} epochs={self.custom_config_dict.get("epochs", 100)} imgsz={self.custom_config_dict.get("imgsz", 640)} batch={self.batch_size} device={self.device} workers={self.workers} project=./Object Detction yolov8_training name={self.model_name}'
        #         os.system(training_command)
        #     except Exception as e:
        #         print(f"Error during YOLOv8 training: {e}")
        #         self.custom_logger.exception(e)
        # else:
        #     # yolo v4
        #     print('Started Creating Repo for local Training of Yolo v-4 model')
        #     os.mkdir('yolov4-main/data/Obj')
        #     print('Completed Creating Repo for local Training of Yolo v-4 model')
        #     print('Started Moving Data for local Training of Yolo v-4 model')
        #     for img_file in os.listdir("training_data/Images/items"):
        #         shutil.copy(f'training_data/Images/items/{img_file}', f'yolov4-main/data/Obj/{img_file}')

        #     for txt_file in os.listdir("training_data/labels/yolo"):
        #         shutil.copy(f'training_data/labels/yolo/{txt_file}', f'yolov4-main/data/Obj/{txt_file}')

        #     print('Completed Moving Data for local Training of Yolo v-4 model')
        #     os.mkdir('./groundtruth_testing_data/data/originalimages')
        #     os.mkdir('./groundtruth_testing_data/data/original Annotation')
        #     os.mkdir('./groundtruth_testing_data/data/original Json')
        #     os.mkdir('./groundtruth_testing_data/data/predicted')
        #     source = './groundtruth_testing_data/data/'
        #     mydict = {
        #         './groundtruth_testing_data/data/originalimages': ['jpg','png','jpeg'],
        #         './groundtruth_testing_data/data/original Annotation': ['txt'],
        #         './groundtruth_testing_data/data/original Json': ['json']
        #     }
        #     for destination, extensions in mydict.items():
        #         for ext in extensions:
        #             for file in glob.glob(source + '*.' + ext):
        #                 print(file)
        #                 shutil.move(file, destination)
        #     from utility.create_list_images import create_train_txt
        #     create_train_txt()
            
        #     # Capture training output and run training
        #     print("Starting YOLOv4 training with MLflow logging...")
        #     training_command = f'./yolov4-main/darknet detector train {self.data_yaml} {self.cfg_yaml} {self.weights}'
            
        #     # Run training and capture output
        #     try:
        #         result = subprocess.run(training_command, shell=True, capture_output=True, text=True, cwd='.')
        #         training_output = result.stdout + result.stderr
        #         print(training_output)  # Still show output in console
                
        #         # Parse training metrics from output
        #         training_metrics = self._parse_training_metrics(training_output)
                
        #     except Exception as e:
        #         print(f"Error during training: {e}")
        #         # Fallback to original method
        #         os.system(training_command)
        #         training_metrics = {}
        #     source_folder = './yolov4-main/backup/'
        #     destination_folder = './model_registry/'
            
        #     # Ensure model_registry directory exists
        #     if not os.path.exists(destination_folder):
        #         os.makedirs(destination_folder)
                
        #     for file_name in os.listdir(source_folder):
        #         # construct full file path with proper path joining
        #         source = os.path.join(source_folder, file_name)
        #         destination = os.path.join(destination_folder, file_name)
        #         # copy only files
        #         if os.path.isfile(source):
        #             shutil.copy(source, destination)
        #             print('Copied Model generated successfully', file_name)
        #     model_list = os.listdir('./model_registry')
        #     print(model_list)
            
        #     # Log training details to MLflow after training completion
        #     try:
        #         import mlflow
        #         import uuid
                
        #         # Set experiment with proper error handling
        #         if self.model_name and self.model_name.strip():
        #             mlflow.set_experiment(self.model_name)
        #         else:
        #             mlflow.set_experiment("YOLOv4_Training2")
                    
        #         # Start MLflow run
        #         with mlflow.start_run():
        #             # Log parameters
        #             mlflow.log_param('Model Name', self.model_name)
        #             mlflow.log_param('YOLO Version', 'YOLOv4')
        #             mlflow.log_param('Batch Size', self.batch_size)
        #             mlflow.log_param('Device', self.device)
        #             mlflow.log_param('Config File', self.cfg_yaml)
        #             mlflow.log_param('Data File', self.data_yaml)
        #             mlflow.log_param('Weights', self.weights)
        #             mlflow.log_param('Hyper Parameter File', self.hyper_parameter_file)
        #             mlflow.log_param('Details of Augmentation', str(self.augmentation_of_data))
        #             mlflow.log_param('Details of Special Augmentation', str(self.special_augmentation_of_data))
        #             mlflow.log_param('Dataset Name', self.dataset_name)
        #             mlflow.log_param('Custom Iteration Id', str(uuid.uuid1()))
        #             mlflow.log_param('Model Registry Path', './model_registry')
        #             mlflow.log_param('Available Models', str(model_list))
                    
        #             # Log training completion status and metrics
        #             mlflow.log_param('Training Status', 'Completed')
        #             mlflow.log_metric('Total Models Generated', len(model_list))
                    
        #             # Log parsed training metrics
        #             for metric_name, metric_value in training_metrics.items():
        #                 if isinstance(metric_value, (int, float)):
        #                     mlflow.log_metric(metric_name, metric_value)
        #                 else:
        #                     mlflow.log_param(metric_name, str(metric_value))
                    
        #             # Log system and training information as metrics
        #             mlflow.log_metric('CPU_Usage_Percent', psutil.cpu_percent())
        #             mlflow.log_metric('Memory_Usage_Percent', psutil.virtual_memory().percent)
        #             mlflow.log_metric('Disk_Usage_Percent', psutil.disk_usage('.').percent)
                    
        #             # Log comprehensive GPU metrics
        #             gpu_metrics, gpu_status = self._get_gpu_metrics()
                    
        #             # Log GPU parameters and metrics
        #             mlflow.log_param('GPU_Detection_Status', gpu_status)
                    
        #             for metric_name, metric_value in gpu_metrics.items():
        #                 if isinstance(metric_value, (int, float)):
        #                     mlflow.log_metric(metric_name, metric_value)
        #                 else:
        #                     mlflow.log_param(metric_name, str(metric_value))
                    
        #             if gpu_metrics:
        #                 print(f"GPU metrics logged: {gpu_status}")
        #             else:
        #                 print(f"No GPU metrics available: {gpu_status}")
                    
        #             # Log only the organized model_registry directory (avoid duplication)
        #             if os.path.exists('./model_registry') and os.listdir('./model_registry'):
        #                 mlflow.log_artifacts('./model_registry', artifact_path="trained_models")
        #                 print(f"Logged {len(model_list)} model files to MLflow under 'trained_models' directory")
                    
        #             # Log training configuration files if they exist
        #             config_files = [self.cfg_yaml, self.data_yaml]
        #             for config_file in config_files:
        #                 if os.path.exists(config_file):
        #                     mlflow.log_artifact(config_file, artifact_path="config")
        #                     print(f"Logged config file: {config_file}")
                    
        #             print("YOLOv4 training details and model artifacts added to MLflow immediately after training completion")
        #             self.custom_logger.info("YOLOv4 training details and model artifacts added to MLflow immediately after training completion")
        #     except Exception as e:
        #         print(f"Failed to log to MLflow: {e}")
        #         self.custom_logger.exception(e)
            
        #     model_value = input("Please Enter Model from above list only on which you want to perform "
        #                         "Groundtruth testing on")
        #     os.system(f'python3 ./yolov4-main/darknet_images.py --input ./groundtruth_testing_data/data/originalimages/ --weights ./yolov4-main/backup/{model_value} --dont_show --save_labels --save_labels_path ./groundtruth_testing_data/data/predicted  --config_file {self.cfg_yaml} --data_file {self.data_yaml} --thresh 0.4')
        #     os.rename('./groundtruth_testing_data/data/originalimages', './groundtruth_testing_data/data/original images')
        #     shutil.copyfile(f'./training_data/labels/{self.dataset_name}.names', f'./groundtruth_testing_data/data/{self.dataset_name}.names')            
            
        #     # Generate comprehensive Ground truth Summary report for YOLOv4
        #     print(f"Generating comprehensive GT reports for YOLOv4")
        #     try:
        #         import pandas as pd
        #         import yaml
        #         import json
        #         from metrics_calculation import calculate_metrics
        #         from generate_overall_summary import export_to_summary_excel
        #         from generate_detailed_gt import generate_detailed_report
                
        #         # Validate required directories exist
        #         required_dirs = [
        #             './groundtruth_testing_data/data/original Annotation/',
        #             './groundtruth_testing_data/data/predicted/',
        #             './groundtruth_testing_data/data/original images/'
        #         ]
                
        #         for dir_path in required_dirs:
        #             if not os.path.exists(dir_path):
        #                 print(f"Error: Required directory not found: {dir_path}")
        #                 raise FileNotFoundError(f"Missing directory: {dir_path}")
        #             if not os.listdir(dir_path):
        #                 print(f"Warning: Directory is empty: {dir_path}")
        #           # Load groundtruth and prediction data from YOLO format files
        #         data_df_path = './groundtruth_testing_data/data/data_df.json' 
        #         if os.path.exists(data_df_path):
        #             try:
        #                 data_df = pd.read_json(data_df_path)
        #                 print(f"Loaded metadata from {data_df_path}: {len(data_df)} entries")
        #                 # Validate required columns exist and add defaults if missing
        #                 required_columns = ['image', 'rawtraining', 'valve']
        #                 for col in required_columns:
        #                     if col not in data_df.columns:
        #                         print(f"Warning: Missing column '{col}' in metadata. Adding default values.")
        #                         data_df[col] = 'unknown'
        #                 # Ensure DataFrame has proper data types and reset index to avoid pandas indexing issues
        #                 data_df = data_df.astype(str).reset_index(drop=True)
        #                 print(f"DataFrame structure validated. Shape: {data_df.shape}")
        #             except Exception as e:
        #                 print(f"Error loading metadata file: {e}. Creating structure from JSON files.")
        #                 data_df = pd.DataFrame({'image': [], 'rawtraining': [], 'valve': []})
        #         else:
        #             print(f"Warning: Metadata file not found at {data_df_path}. Creating structure from JSON files.")
        #             # Create metadata structure from individual JSON files in "original Json" folder
        #             json_metadata_dir = './groundtruth_testing_data/data/original Json/'
        #             metadata_records = []
                    
        #             if os.path.exists(json_metadata_dir):
        #                 json_files = [f for f in os.listdir(json_metadata_dir) if f.endswith('.json')]
        #                 print(f"Found {len(json_files)} JSON metadata files")
                        
        #                 for json_file in json_files:
        #                     try:
        #                         json_path = os.path.join(json_metadata_dir, json_file)
        #                         with open(json_path, 'r') as f:
        #                             json_data = json.load(f)
                                
        #                         if 'user' in json_data:
        #                             user_data = json_data['user']
        #                             # Extract image name from JSON filename
        #                             image_name = os.path.splitext(json_file)[0]
                                    
        #                             # Create metadata record
        #                             metadata_record = {
        #                                 'image': image_name,
        #                                 'rawtraining': user_data.get('rawtraining', 'unknown'),
        #                                 'valve': user_data.get('valve', user_data.get('valve', 'unknown')),  # Handle case variations
        #                                 #'lighting': user_data.get('lighting', 'unknown'),
        #                                 #'distance': user_data.get('distance', user_data.get('Distance', 'unknown'))  # Handle case variations
        #                             }
        #                             metadata_records.append(metadata_record)
        #                             print(f"Loaded metadata for {image_name}: {metadata_record}")
        #                         else:
        #                             print(f"Warning: No 'user' key in {json_file}")
        #                     except Exception as e:
        #                         print(f"Warning: Error reading {json_file}: {e}")
        #                         continue
                    
        #             if metadata_records:
        #                 data_df = pd.DataFrame(metadata_records)
        #                 print(f"Successfully created metadata DataFrame from JSON files: {len(data_df)} entries")
        #             else:
        #                 print("No valid JSON metadata found. Using empty structure.")
        #                 data_df = pd.DataFrame({'image': [], 'rawtraining': [], 'valve': []})
                
        #         # Parse ground truth data
        #         groundtruth_details = []
        #         gt_annotation_path = './groundtruth_testing_data/data/original Annotation/'
        #         gt_files = [f for f in os.listdir(gt_annotation_path) if f.endswith('.txt')]
        #         print(f"Found {len(gt_files)} ground truth annotation files")
                
        #         for annotation_file in gt_files:
        #             image_name = annotation_file.replace('.txt', '')
        #             file_path = os.path.join(gt_annotation_path, annotation_file)
        #             with open(file_path, 'r') as f:
        #                 lines = f.readlines()
        #                 for line_num, line in enumerate(lines, 1):
        #                     line = line.strip()
        #                     if not line:  # Skip empty lines
        #                         continue
        #                     parts = line.split()
        #                     if len(parts) >= 5:
        #                         try:
        #                             class_id, x_center, y_center, width, height = map(float, parts[:5])
        #                             class_id = int(class_id)
                                    
        #                             # Validate class ID is within expected range (0-4 for 5 valve classes)
        #                             if not (0 <= class_id <= 4):
        #                                 print(f"Warning: Invalid class ID {class_id} in {annotation_file} line {line_num} - skipping (prevents background class)")
        #                                 continue
                                    
        #                             class_name = get_class_name(class_id)
        #                             if class_name is None:
        #                                 print(f"Warning: Could not map class ID {class_id} in {annotation_file} line {line_num} - skipping")
        #                                 continue
                                    
        #                             groundtruth_details.append({
        #                                 'image_name': image_name,
        #                                 'class_name': class_name,  # Use actual class name, not generic class_N
        #                                 'bounding_box': [x_center, y_center, width, height]
        #                             })
        #                         except ValueError as e:
        #                             print(f"Error parsing GT file {annotation_file}, line {line_num}: {line} - {e}")
        #                             continue
        #                     else:
        #                         print(f"Warning: Invalid GT annotation format in {annotation_file}, line {line_num}: {line}")
                
        #         print(f"Parsed {len(groundtruth_details)} ground truth annotations")
                
        #         # Parse prediction data
        #         prediction_details = []
        #         pred_annotation_path = './groundtruth_testing_data/data/predicted/'
        #         pred_files = [f for f in os.listdir(pred_annotation_path) if f.endswith('.txt')]
        #         print(f"Found {len(pred_files)} prediction files")
                
        #         for prediction_file in pred_files:
        #             image_name = prediction_file.replace('.txt', '')
        #             file_path = os.path.join(pred_annotation_path, prediction_file)
        #             with open(file_path, 'r') as f:
        #                 lines = f.readlines()
        #                 for line_num, line in enumerate(lines, 1):
        #                     line = line.strip()
        #                     if not line:  # Skip empty lines
        #                         continue
        #                     parts = line.split()
        #                     if len(parts) >= 6:
        #                         try:
        #                             from class_mapping import is_valid_class_id, get_class_name
                                    
        #                             class_id, confidence, x_center, y_center, width, height = map(float, parts[:6])
        #                             prediction_details.append({
        #                                 'image_name': image_name,
        #                                 'class_name': f'class_{int(class_id)}',
        #                                 'bounding_box': [x_center, y_center, width, height],
        #                                 'confidence_score': confidence
        #                             })
        #                         except ValueError as e:
        #                             print(f"Error parsing prediction file {prediction_file}, line {line_num}: {line} - {e}")
        #                             continue
        #                     else:
        #                         print(f"Warning: Invalid prediction format in {prediction_file}, line {line_num}: {line}")
                
        #         print(f"Parsed {len(prediction_details)} prediction annotations")
                
        #         # Validate we have data to process
        #         if not groundtruth_details:
        #             print("Error: No valid ground truth annotations found!")
        #             print("Please check that:")
        #             print("1. Ground truth annotation files exist in ./groundtruth_testing_data/data/original Annotation/")
        #             print("2. Files are in YOLO format: class_id x_center y_center width height")
        #             print("3. Files contain valid numerical data")
        #             raise ValueError("No ground truth data available for processing")
                
        #         if not prediction_details:
        #             print("Warning: No valid prediction annotations found!")
        #             print("This may indicate:")
        #             print("1. Model made no predictions above confidence threshold")
        #             print("2. Prediction files are empty or in wrong format")
        #             print("3. YOLOv4 inference failed")
        #             print("Proceeding with empty predictions (all ground truth will be missed detections)")
                
        #         # Generate comprehensive metrics and Excel reports
        #         iou_threshold = 0.5
        #         print(f"Starting metrics calculation with IOU threshold: {iou_threshold}")
        #         report_folder = calculate_metrics(groundtruth_details, prediction_details, data_df, iou_threshold)
        #         print(f"Metrics calculation completed. Reports saved to: {report_folder}")
                
        #         # Generate overall summary Excel with safe calculations
        #         if groundtruth_details:
        #             # Calculate summary metrics for overall summary Excel
        #             tp = 0
        #             if prediction_details:
        #                 tp = len([p for p in prediction_details if any(abs(p['bounding_box'][0] - gt['bounding_box'][0]) < 0.1 for gt in groundtruth_details)])
                    
        #             fp = len(prediction_details) - tp if prediction_details else 0
        #             fn = len(groundtruth_details) - tp
                    
        #             overall_summary_data = {
        #                 'overall_metrics': {
        #                     'data size': len(groundtruth_details),
        #                     'tp': tp,
        #                     'fp': fp,
        #                     'fn': fn,
        #                     'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        #                     'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        #                     'f1': (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        #                 },
        #                 'class_wise_metrics': {},
        #                 'category_wise_metrics': {},
        #                 'class_category_wise_metrics': {}
        #             }
                    
        #             print("Generating overall summary Excel report...")
        #             export_to_summary_excel(overall_summary_data, report_folder)
                
        #         # Generate detailed report with images and annotations
        #         if os.path.exists('./groundtruth_testing_data/data/original images/'):
        #             print("Generating detailed report with images...")
        #             # Use actual class names from the data
        #             unique_classes = list(set(gt['class_name'] for gt in groundtruth_details))
        #             tag_list = unique_classes if unique_classes else [f'class_{i}' for i in range(10)]
        #             try:
        #                 generate_detailed_report(
        #                     report_path=report_folder,
        #                     images_path='./groundtruth_testing_data/data/original images/',
        #                     annotation_path='./groundtruth_testing_data/data/original Annotation/',
        #                     predicted_annotation_path='./groundtruth_testing_data/data/predicted/',
        #                     metadata_path='./groundtruth_testing_data/data/original Json/',
        #                     tag_list=tag_list,
        #                     inference_image_path=os.path.join(report_folder, 'inference_images')
        #                 )
        #                 print("Detailed report generation completed")
        #             except Exception as detail_error:
        #                 print(f"Warning: Detailed report generation failed: {detail_error}")
        #                 print("Continuing with other reports...")
                
        #         print(f"Generated comprehensive GT reports for YOLOv4 in: {report_folder}")
                
        #         # Copy summary report to gt_reports for compatibility
        #         if not os.path.exists('gt_reports'):
        #             os.makedirs('gt_reports')
                
        #         # Look for the main summary file from metrics_calculation
        #         summary_candidates = [
        #             os.path.join(report_folder, 'Detection_Summary.xlsx'),
        #             os.path.join(report_folder, 'Overall_summary.xlsx'),
        #             os.path.join(report_folder, 'overall_summary.xlsx')
        #         ]
                
        #         for summary_excel in summary_candidates:
        #             if os.path.exists(summary_excel):
        #                 shutil.copy2(summary_excel, 'gt_reports/class_wise_report.xlsx')
        #                 print(f"Copied {summary_excel} to gt_reports/class_wise_report.xlsx")
        #                 break
        #         else:
        #             print("Warning: No summary Excel file found to copy to gt_reports/")
        #             shutil.copy2(summary_excel, 'gt_reports/class_wise_report.xlsx')
                
        #         # Log GT testing details to MLflow after GT report generation
        #         try:
        #             # Set experiment with proper error handling
        #             gt_experiment_name = f"{self.model_name}_GT_Testing" if self.model_name and self.model_name.strip() else "YOLOv4_GT_Testing"
        #             mlflow.set_experiment(gt_experiment_name)
                        
        #             # Start MLflow run for GT testing
        #             with mlflow.start_run():
        #                 # Log GT testing parameters
        #                 mlflow.log_param('Model Used for GT Testing', model_value)
        #                 mlflow.log_param('Model Name', self.model_name)
        #                 mlflow.log_param('YOLO Version', 'YOLOv4')
        #                 mlflow.log_param('GT Testing Status', 'Completed')
        #                 mlflow.log_param('Dataset Name', self.dataset_name)
        #                 mlflow.log_param('Config File', self.cfg_yaml)
        #                 mlflow.log_param('Data File', self.data_yaml)
        #                 mlflow.log_param('GT Testing Threshold', 0.4)
        #                 mlflow.log_param('GT Data Path', './groundtruth_testing_data/data')
        #                 mlflow.log_param('GT Report Config', str(self.gt_report_config_dict))
        #                 mlflow.log_param('Custom GT Testing Id', str(uuid.uuid1()))
                        
        #                 # Log system metrics during GT testing
        #                 mlflow.log_metric('CPU_Usage_Percent', psutil.cpu_percent())
        #                 mlflow.log_metric('Memory_Usage_Percent', psutil.virtual_memory().percent)
        #                 mlflow.log_metric('Disk_Usage_Percent', psutil.disk_usage('.').percent)
                        
        #                 # Log comprehensive GPU metrics for GT testing
        #                 gpu_metrics, gpu_status = self._get_gpu_metrics()
                        
        #                 # Log GPU parameters and metrics
        #                 mlflow.log_param('GPU_Detection_Status', gpu_status)
                        
        #                 for metric_name, metric_value in gpu_metrics.items():
        #                     if isinstance(metric_value, (int, float)):
        #                         mlflow.log_metric(f"GT_{metric_name}", metric_value)
        #                     else:
        #                         mlflow.log_param(f"GT_{metric_name}", str(metric_value))
                        
        #                 # Log GT report artifacts
        #                 if os.path.exists('gt_reports/class_wise_report.xlsx'):
        #                     mlflow.log_artifact('gt_reports/class_wise_report.xlsx', artifact_path="gt_reports")
        #                     print("Logged GT class_wise_report.xlsx to MLflow")
                        
        #                 # Log original GT data structure info
        #                 if os.path.exists('./groundtruth_testing_data/data'):
        #                     gt_images = len([f for f in os.listdir('./groundtruth_testing_data/data/original images') if f.endswith(('.png', '.jpg', '.jpeg'))])
        #                     gt_annotations = len([f for f in os.listdir('./groundtruth_testing_data/data/original Annotation') if f.endswith('.txt')])
        #                     gt_predictions = len([f for f in os.listdir('./groundtruth_testing_data/data/predicted') if f.endswith('.txt')])
                            
        #                     mlflow.log_metric('GT_Total_Images', gt_images)
        #                     mlflow.log_metric('GT_Total_Annotations', gt_annotations)
        #                     mlflow.log_metric('GT_Total_Predictions', gt_predictions)
                        
        #                 print("YOLOv4 GT testing details and report artifacts added to MLflow after GT testing completion")
        #                 self.custom_logger.info("YOLOv4 GT testing details and report artifacts added to MLflow after GT testing completion")
                        
        #                 # Generate summary charts and upload to MLflow after all Excel reports are completed
        #                 print("Generating summary charts and uploading to MLflow...")
        #                 try:
        #                     from generate_summary_representation import generate_summary_graph
                            
        #                     # Create config structure required by summary_report.py
        #                     val_config = {
        #                         'report_generator': {
        #                             'summary_header': ['valve']  # Default headers for summary report
        #                         }
        #                     }
                            
        #                     # Generate summary graphs and charts
        #                     generate_summary_graph(report_folder, val_config)
        #                     print(f"Summary charts generated successfully in {report_folder}/Merged_Comparison_Charts")
                            
        #                     # Upload charts to current MLflow run since we're already in a run
        #                     try:
        #                         chart_folder = os.path.join(report_folder, "Merged_Comparison_Charts")
        #                         if os.path.exists(chart_folder):
        #                             for chart_file in os.listdir(chart_folder):
        #                                 if chart_file.endswith('.png'):
        #                                     chart_path = os.path.join(chart_folder, chart_file)
        #                                     mlflow.log_artifact(chart_path, "summary_charts")
        #                                     print(f"Uploaded chart {chart_file} to MLflow")
                                    
        #                             # Log chart generation metrics
        #                             mlflow.log_param("chart_generation_date", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        #                             mlflow.log_param("yolo_version", "YOLOv4")
        #                             mlflow.log_param("model_used", model_value)
        #                             mlflow.log_metric("total_charts_generated", len([f for f in os.listdir(chart_folder) if f.endswith('.png')]))
                                        
        #                             print("All summary charts uploaded to MLflow successfully!")
        #                         else:
        #                             print(f"Chart folder not found at {chart_folder}")
        #                     except Exception as mlflow_chart_error:
        #                         print(f"Warning: Failed to upload charts to MLflow: {mlflow_chart_error}")
                                
        #                 except Exception as chart_error:
        #                     print(f"Warning: Failed to generate summary charts: {chart_error}")
        #                     print("Detailed error:")
        #                     traceback.print_exc()
                        
        #         except Exception as e:
        #             print(f"Failed to log GT testing to MLflow: {e}")
        #             self.custom_logger.exception(e)
                    
        #     except Exception as e:
        #         print(e)
    
    # def _create_validation_split_for_yolov8(self):
    #     """Create validation split from training data for YOLOv8"""
    #     import random
        
    #     # Get training images and labels
    #     train_images = os.listdir("training_data/Images/items")
    #     train_labels = os.listdir("training_data/labels/yolo")
        
    #     # Create a mapping of image to label files
    #     image_label_pairs = []
    #     for img_file in train_images:
    #         img_name = os.path.splitext(img_file)[0]
    #         label_file = f"{img_name}.txt"
    #         if label_file in train_labels:
    #             image_label_pairs.append((img_file, label_file))
        
    #     # Split 20% for validation
    #     random.shuffle(image_label_pairs)
    #     split_point = int(len(image_label_pairs) * 0.8)
        
    #     train_pairs = image_label_pairs[:split_point]
    #     val_pairs = image_label_pairs[split_point:]
        
    #     print(f"Creating validation split: {len(train_pairs)} training, {len(val_pairs)} validation samples")
        
    #     # Copy validation files
    #     for img_file, label_file in val_pairs:
    #         shutil.copy(f'training_data/Images/items/{img_file}', f'yolov8-main/data/val/images/{img_file}')
    #         shutil.copy(f'training_data/labels/yolo/{label_file}', f'yolov8-main/data/val/labels/{label_file}')
        
    #     return val_pairs
    
    # def _populate_gt_testing_data_from_validation(self):
    #     """Populate ground truth testing data from validation split"""
    #     # Ensure GT testing directories exist
    #     os.makedirs('./groundtruth_testing_data/data', exist_ok=True)
        
    #     # Copy validation data to GT testing directory for later testing
    #     val_images_dir = 'yolov8-main/data/val/images'
    #     val_labels_dir = 'yolov8-main/data/val/labels'
        
    #     if os.path.exists(val_images_dir):
    #         for img_file in os.listdir(val_images_dir):
    #             if img_file.endswith(('.png', '.jpg', '.jpeg')):
    #                 shutil.copy(f'{val_images_dir}/{img_file}', f'./groundtruth_testing_data/data/{img_file}')
        
    #     if os.path.exists(val_labels_dir):
    #         for label_file in os.listdir(val_labels_dir):
    #             if label_file.endswith('.txt'):
    #                 shutil.copy(f'{val_labels_dir}/{label_file}', f'./groundtruth_testing_data/data/{label_file}')
        
    #     print("Populated ground truth testing data from validation split")
    
    # def _convert_yolo_predictions_to_absolute(self, predicted_dir, images_dir):
    #     """Convert YOLO format predictions to absolute coordinates format expected by gt_inference"""
    #     import cv2
        
    #     # Check if predictions are in labels subdirectory (from command line inference)
    #     labels_subdir = os.path.join(predicted_dir, 'labels')
    #     source_dir = predicted_dir
    #     if os.path.exists(labels_subdir):
    #         source_dir = labels_subdir
        
    #     for pred_file in os.listdir(source_dir):
    #         if pred_file.endswith('.txt'):
    #             # Find corresponding image
    #             image_name = os.path.splitext(pred_file)[0]
    #             image_path = None
    #             for ext in ['jpg', 'jpeg', 'png']:
    #                 potential_path = os.path.join(images_dir, f'{image_name}.{ext}')
    #                 if os.path.exists(potential_path):
    #                     image_path = potential_path
    #                     break
                
    #             if image_path is None:
    #                 print(f"Warning: Could not find image for {pred_file}")
    #                 continue
                
    #             # Read image to get dimensions
    #             img = cv2.imread(image_path)
    #             if img is None:
    #                 print(f"Warning: Could not read image {image_path}")
    #                 continue
                
    #             img_height, img_width = img.shape[:2]
                
    #             # Read YOLO format predictions
    #             pred_path = os.path.join(source_dir, pred_file)
    #             output_path = os.path.join(predicted_dir, pred_file)
                
    #             with open(pred_path, 'r') as f_in, open(output_path, 'w') as f_out:
    #                 for line in f_in:
    #                     parts = line.strip().split()
    #                     if len(parts) >= 5:
    #                         class_id = int(parts[0])
    #                         x_center = float(parts[1])
    #                         y_center = float(parts[2])
    #                         width = float(parts[3])
    #                         height = float(parts[4])
    #                         confidence = float(parts[5]) if len(parts) > 5 else 0.5
                            
    #                         # Convert from normalized YOLO to absolute coordinates
    #                         x1 = (x_center - width/2) * img_width
    #                         y1 = (y_center - height/2) * img_height
    #                         x2 = (x_center + width/2) * img_width
    #                         y2 = (y_center + height/2) * img_height
                            
    #                         # Write in absolute coordinate format (class_id x1 y1 x2 y2 confidence)
    #                         f_out.write(f"{class_id} {confidence:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} \n")
        
    #     # Clean up the labels subdirectory if it exists
    #     if os.path.exists(labels_subdir):
    #         shutil.rmtree(labels_subdir)
    #         print("Cleaned up labels subdirectory after conversion")
    
    # def _validate_and_fix_gt_annotations(self, annotations_dir):
    #     """Validate and fix ground truth annotation files to ensure proper YOLO format"""
    #     print(f"Validating ground truth annotations in {annotations_dir}")
        
    #     for annotation_file in os.listdir(annotations_dir):
    #         if annotation_file.endswith('.txt'):
    #             file_path = os.path.join(annotations_dir, annotation_file)
    #             print(f"Checking {annotation_file}")
                
    #             # Read and validate each line
    #             valid_lines = []
    #             with open(file_path, 'r') as f:
    #                 for line_num, line in enumerate(f, 1):
    #                     line = line.strip()
    #                     if not line:
    #                         continue
                            
    #                     parts = line.split()
    #                     if len(parts) < 5:
    #                         print(f"Warning: {annotation_file} line {line_num} has insufficient parts: {line}")
    #                         continue
                        
    #                     try:
    #                         # Validate class_id is integer
    #                         class_id = int(float(parts[0]))  # Use float first in case it's "0.0"
                            
    #                         # Validate coordinates are floats between 0 and 1
    #                         x_center = float(parts[1])
    #                         y_center = float(parts[2])
    #                         width = float(parts[3])
    #                         height = float(parts[4])
                            
    #                         # Ensure coordinates are normalized (between 0 and 1)
    #                         if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
    #                             print(f"Warning: {annotation_file} line {line_num} has non-normalized coordinates: {line}")
    #                             # Try to normalize if values seem to be absolute coordinates
    #                             if x_center > 1 or y_center > 1 or width > 1 or height > 1:
    #                                 print(f"Attempting to normalize coordinates for {annotation_file} line {line_num}")
    #                                 # This is a rough normalization - ideally we'd need image dimensions
    #                                 # For now, skip this line or handle it manually
    #                                 continue
                            
    #                         # Reconstruct the line with proper format
    #                         valid_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                            
    #                     except ValueError as e:
    #                         print(f"Error: {annotation_file} line {line_num} has invalid format: {line} - {e}")
    #                         continue
                
    #             # Write back the validated lines
    #             if valid_lines:
    #                 with open(file_path, 'w') as f:
    #                     for line in valid_lines:
    #                         f.write(line + '\n')
    #                 print(f"Fixed {annotation_file}: {len(valid_lines)} valid annotations")
    #             else:
    #                 print(f"Warning: No valid annotations found in {annotation_file}")
        
    #     print("Ground truth annotation validation completed")
            
    def clean_resources(self):
        # Delete all unwanted files
        try:
            if self.custom_pipeline_yolo_version == "YoloV7":
                os.chmod('training_data', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                shutil.rmtree('training_data')
                shutil.rmtree('yolov7-main/data/train')
                shutil.rmtree('yolov7-main/data/val')
            elif self.custom_pipeline_yolo_version == "YoloV8":
                os.chmod('training_data', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                shutil.rmtree('training_data')
                shutil.rmtree('yolov8-main/data/train')
                shutil.rmtree('yolov8-main/data/val')
                # Optionally clean up training results
                if os.path.exists('yolov8_training'):
                    shutil.rmtree('yolov8_training')
                # Clean up YOLOv8 GT testing artifacts
                if os.path.exists('gt_reports'):
                    shutil.rmtree('gt_reports')
            else:
                os.chmod('training_data', stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
                shutil.rmtree('training_data')
                shutil.rmtree('yolov4-main/data/Obj')
                shutil.rmtree('yolov4-main/data/val')
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)
        try:
            shutil.rmtree("groundtruth_testing_data")
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)


# YOLOv7 specific class  
class CustomYoloPipelineYoloV7(CustomYoloPipelineYoloV4):
    """ Custom Training Pipeline with YOLOv7 model - inherits from YoloV4 class """
    
    def __init__(self, custom_config_dict, custom_logger, gt_report_config_dict, custom_pipeline_yolo_version):
        # Call parent constructor with YoloV7 version
        super().__init__(custom_config_dict, custom_logger, gt_report_config_dict, "YoloV7")


# YOLOv8 specific class
class CustomYoloPipelineYoloV8(CustomYoloPipelineYoloV4):
    """ Custom Training Pipeline with YOLOv8 model - inherits from YoloV4 class """
    
    def __init__(self, custom_config_dict, custom_logger, gt_report_config_dict, custom_pipeline_yolo_version):
        # Call parent constructor with YoloV8 version
        super().__init__(custom_config_dict, custom_logger, gt_report_config_dict, "YoloV8")
