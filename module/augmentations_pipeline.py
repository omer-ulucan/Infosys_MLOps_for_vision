'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import cv2
import random
import warnings
import numpy as np
from PIL import Image
import albumentations as A
import os
import re
import glob
import utility.constants as const
import shutil

warnings.filterwarnings("ignore")



class AugmentationPipeline:
        

    def __init__(self, logger, augmentation_dict, gt_report_dict):
        self.custom_logger = logger
        self.val_config = augmentation_dict
        self.input_from = self.val_config['path']['input_from']
        self.output_to = self.val_config['path']['output_to']
        self.dataset_name = self.val_config["labels_file_name"]
        self.image_extensions = self.val_config["image_extensions"]
        self.tag_list = [line.strip() for line in open(os.path.join(self.input_from, self.dataset_name)+'.names', 'r')]
        self.augmentation_techniques = self.val_config["generic_augmentation"]["augmentation_techniques"]
        self.augmentation_type = self.val_config["generic_augmentation"]["augmentation_type"]
        self.special_augmentation_techniques = self.val_config["special_augmentation"]["augmentation_techniques"]
        self.class_list = self.val_config["special_augmentation"]["class_list"]
        self.save_original_file = self.val_config["save_original_file"]
        

        self.transforms = None
        self.special_transforms = None
        self.length_augmentation_list = None
        self.labels_file_path = None


    def perform_augmentation(self):

        """ Perform Special Augmentation on user provided classes and generic Augmentation on all classes"""
        print("Current Iteration has following classes : ", self.tag_list)
        print()
        
        try:
            special_class_index_list = []
            if self.class_list:
                for special_class in self.class_list:
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

        def write_image_box(image_write, bboxes_write, filename_write):
            """ Generate Augmented images and text files"""
            cv2.imwrite(os.path.join(self.output_to , filename_write) , image_write[:, :, ::-1])
            with open(os.path.join(self.output_to , os.path.splitext(filename_write)[0]) + ".txt", 'w+') as file:
                for box in bboxes_write:
                    x, y, w, h, category = box
                    file.write(f"{int(category)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            file.close()

        import os

        def save_original_files(input_image_path, output_to, input_from):
            
            relative_path = os.path.relpath(input_image_path, start=input_from)
            output_path = os.path.join(output_to,"original_files", relative_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(input_image_path, output_path)

        # IMPORT ONLY IN YOLO FORMAT

        # SPECIAL AUGMENTATION on User specific classes
        try:
            # Special data Augmentations performed or not on user specified class
            # See a list of transforms performed by Albumentations.
            # https://albumentations.ai/docs/getting_started/transforms_and_targets/
            if self.special_augmentation_techniques and self.class_list:
                print('Performing Special Augmentations on classes specified by the User')
                self.custom_logger.info('Performing Special Augmentations on classes specified by the User')
                self.special_transforms = []
                for special_aug_item in self.special_augmentation_techniques:
                    for special_aug_key, special_aug_val in special_aug_item.items():
                        special_aug_val["always_apply"] = True
                        special_aug_data = [getattr(A, special_aug_key)(**special_aug_val)]
                        self.special_transforms.append(special_aug_data)
                print("Special Aug list", self.special_transforms)
                self.custom_logger.info(f"Special Aug list {self.special_transforms}")

                self.labels_file_path = glob.glob(self.input_from + '/**/*.txt', recursive=True)

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

                special_aug_images = list(map(do_split, special_aug_files))

                for special_images in special_aug_images:
                    try:                        
                        all_files = []
                        all_file_paths = []
                        
                        for root, dirs, files in os.walk(self.input_from):
                            for file in files:
                                all_files.append(file)
                                all_file_paths.append(os.path.join(root, file))

                        r = re.compile(f"{special_images}.(png|jpg)")
                        mapped_img = list(filter(r.match, all_files))
                        file_name = None  # Default if not found
                        annotation_path = None
                        for full_path in all_file_paths:
                            if os.path.basename(full_path) == mapped_img[0]:
                                file_name = full_path
                                annotation_path = os.path.splitext(file_name)[0] + '.txt'
                                break
                        if mapped_img:
                            working_img = mapped_img[0]
                            special_filename = os.path.splitext(working_img)[0]
                            special_image = cv2.imread(file_name)
                            special_image = cv2.cvtColor(special_image, cv2.COLOR_BGR2RGB)
                            
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
                                transform_name = special_aug[0].__class__.__name__
                                working_img, ext = os.path.splitext(working_img)
                                write_image_box(special_res["image"], special_res["bboxes"], working_img + '_' + transform_name.lower() + ext)
                                # write_image_box(special_res["image"], special_res["bboxes"], working_img + '_' + transform_name.lower())
                                if self.save_original_file == True:
                                        save_original_files(file_name, self.output_to, self.input_from)
                                        save_original_files(annotation_path, self.output_to, self.input_from)

                                        
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
            if self.augmentation_techniques:
                self.transforms = []
                for aug_item in self.augmentation_techniques:
                    for aug_key, aug_val in aug_item.items():
                        aug_val["always_apply"] = True
                        aug_data = [getattr(A, aug_key)(**aug_val)]
                        self.transforms.append(aug_data)
                print("Generic Aug list", self.transforms)
                self.custom_logger.info(f"Generic Aug list{self.transforms}")

                if self.augmentation_type == const.FULL:
                    print('Performing Generic Standard Augmentations on all classes  as specified by the User')
                    self.custom_logger.info('Performing Generic Standard Augmentations on all classes  as specified by '
                                            'the User')
                    
                    for folder in os.listdir(self.input_from):
                        folder_path = os.path.join(self.input_from, folder)
                        if os.path.isdir(folder_path):
                            try:
                                for file in os.listdir(folder_path):
                                    if file.endswith(tuple(self.image_extensions)):
                                        filename = os.path.splitext(file)[0]
                                        # print(os.path.abspath(file))
                                        image_path = os.path.join(folder_path , file)
                                        image = cv2.imread(image_path)
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                                        annotation_path = os.path.join(folder_path , str(filename) )+ ".txt"
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
                                            transform_name = aug[0].__class__.__name__
                                            working_img, ext = os.path.splitext(file)
                                            write_image_box(res["image"], res["bboxes"], working_img + '_' + transform_name.lower() + ext)
                                            # write_image_box(res["image"], res["bboxes"], file +'_'+transform_name.lower())
                                            if self.save_original_file == True:
                                                save_original_files(image_path, self.output_to, self.input_from)
                                                save_original_files(annotation_path, self.output_to, self.input_from)
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
                    # self.labels_file_path = glob.glob(self.input_from + '*.txt')
                    self.labels_file_path = glob.glob(os.path.join(self.input_from, '**', '*.txt'), recursive=True)


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
                                
                                all_files = []
                                all_file_paths = []
                                for root, dirs, files in os.walk(self.input_from):
                                    for file in files:
                                        all_files.append(file)
                                        all_file_paths.append(os.path.join(root, file))

                                r = re.compile(f"{image_filename}.(png|jpg)")
                                mapped_img = list(filter(r.match, all_files))
                                if mapped_img:
                                    try:    
                                        file_name = None  # Default if not found
                                        annotation_path = None
                                        for full_path in all_file_paths:
                                            if os.path.basename(full_path) == mapped_img[0]:
                                                file_name = full_path
                                                annotation_path = os.path.splitext(file_name)[0] + '.txt'
                                                break


                                        image = cv2.imread(file_name)
                                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
                                        transform_name = self.transforms[image_chunk_count][0].__class__.__name__
                                        working_img, ext = os.path.splitext(mapped_img[0])
                                        write_image_box(res["image"], res["bboxes"], working_img + '_' + transform_name.lower() + ext)
                                        # write_image_box(res["image"], res["bboxes"], (mapped_img[0]) +'_'+transform_name.lower())
                                        if self.save_original_file == True:
                                                save_original_files(file_name, self.output_to, self.input_from)
                                                save_original_files(annotation_path, self.output_to, self.input_from)
                                    
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