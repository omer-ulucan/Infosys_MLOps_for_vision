'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import os
import random
import shutil
import warnings
import utility.constants as const 
from utility.check_img_type import identify_category
from utility.json_ops import create_json
warnings.simplefilter(action='ignore', category=FutureWarning)



class Local:
    def __init__(self, raw_data_config_dict, custom_logger, pipeline_type):
        self.custom_logger = custom_logger
        self.raw_data_config_dict = raw_data_config_dict
        self.pipeline_type = pipeline_type
        self.img_path_data = self.raw_data_config_dict.get('raw_img_path')
        self.original_data_path = self.raw_data_config_dict.get('original_data_path')
        self.json_label_path_data = self.raw_data_config_dict.get('raw_json_label_path')
        self.ground_truth_percentage = self.raw_data_config_dict.get('ground_truth_percentage')
        self.img_path = None
        self.json_label_path = None
        self.msg = None
        self.tag_list = None
        self.data_type = None
        if self.pipeline_type == const.TRAINING_PIPELINE:
            if os.path.exists("training_data"):
                shutil.rmtree("training_data")
            else:
                pass
            os.mkdir('training_data')
            os.mkdir('training_data/Images/')
            os.mkdir('training_data/Images/items')
            os.mkdir('training_data/labels/')
            os.mkdir('training_data/labels/yolo')
            os.mkdir('training_data/labels/json')
        else:
            pass
        if os.path.exists("json_data"):
            shutil.rmtree("json_data")
        else:
            pass
        os.mkdir('json_data')

        # Refreshing Ground truth dataset
        if os.path.exists("groundtruth_testing_data"):
            shutil.rmtree("groundtruth_testing_data")
        else:
            pass
        os.mkdir('groundtruth_testing_data')
        os.mkdir('groundtruth_testing_data/data')
        if self.pipeline_type == const.TRAINING_PIPELINE:
            pass
        else:
            os.mkdir('groundtruth_testing_data/data/original images')
            os.mkdir('groundtruth_testing_data/data/original Annotation')
            os.mkdir('groundtruth_testing_data/data/original Json')
            os.mkdir('groundtruth_testing_data/data/predicted')


    def raw_data_operation(self):
        # Generating json meta data JSON files for raw data from folder structure
        try:
            for path_val, sub_dirs_val, files_val in os.walk(self.original_data_path):
                for name in files_val:
                    if name.endswith('.txt'):
                        meta_data = os.path.normpath(path_val).split(os.path.sep)
                        meta_data_dict = dict((x.strip(), y.strip())
                                              for x, y in (element.split('-')
                                                           for element in meta_data))
                        final_name = os.path.splitext(name)[0]
                        create_json("json_data", final_name, meta_data_dict)
                    else:
                        pass
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # TrainingPipeline raw data operations
        if self.pipeline_type == const.TRAINING_PIPELINE:
            # Analysis of Raw data for Ground truth type selection
            try:
                categories = {'SO': [], 'MO': [], 'MOC': []}
                path = self.original_data_path
                self.data_type = identify_category(path, categories)
                # print(self.data_type)
                # print(self.data_type['SO'])
                # print(self.data_type['MO'])
                # print(self.data_type['MOC'])
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Check whether MO,SO or MOC and decide Ground truth strategy
            # MOC GT Strategy
            if len(self.data_type['MOC']) > 0:
                # Transferring images for ground truth for MOC
                try:
                    moc_ground_truth_data = []
                    for path_val, sub_dirs_val, files_val in os.walk(self.original_data_path):
                        for name in files_val:
                            if name.endswith(('.jpg', '.png')):
                                meta_data_val = os.path.join(path_val, name)
                                moc_ground_truth_data.append(meta_data_val)
                            else:
                                pass
                    moc_gt_data = len(moc_ground_truth_data)
                    moc_gt_data_count = round(moc_gt_data * self.ground_truth_percentage)
                    file_name = random.sample(moc_ground_truth_data, moc_gt_data_count)
                    print(moc_gt_data, moc_gt_data_count, moc_ground_truth_data)
                    for fname_image in file_name:
                        shutil.move(fname_image, 'groundtruth_testing_data/data')
                    print("Generated ground truth Data class vise")
                    self.custom_logger.info("Generated ground truth Data class vise")
                except Exception as e:
                    print(e)
                    self.custom_logger.exception(e)

            # MO SO GT Strategy
            else:
                # Transferring images for ground truth for SO and MO
                try:
                    gt_data = [os.path.join(path) for path, subdirs, files in
                               os.walk(self.original_data_path) for name in files]
                    gt_data = set(gt_data)
                    data_description = dict();
                    for i in gt_data:
                        _, _, files = next(os.walk(i))
                        files_final = [i for i in files if i.endswith(('.jpg', '.png'))]
                        file_count = len(files_final)
                        i = i.replace('\\', '/')
                        print(file_count)
                        data_description[i] = round(file_count * self.ground_truth_percentage)
                    print(data_description)
                    for key, val in data_description.items():
                        dirpath = key
                        _, _, files_val = next(os.walk(dirpath))
                        files_final = [i for i in files_val if i.endswith(('.jpg', '.png'))]
                        file_name = random.sample(files_final, val)
                        for fname_image in file_name:
                            src_path = os.path.join(dirpath, fname_image)
                            shutil.move(src_path, 'groundtruth_testing_data/data')
                    print("Generated ground truth Data class vise")
                    self.custom_logger.info("Generated ground truth Data class vise")
                except Exception as e:
                    print("Could Not generate ground truth Data class vise")
                    self.custom_logger.exception("Could Not generate ground truth Data class vise")

            # Images to root of training_data
            try:
                source = self.original_data_path
                destination = self.img_path_data + "/" + 'items'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith(('jpg', 'png')):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Labels to root of training_data
            try:
                source = self.original_data_path
                destination = self.json_label_path_data + "/" + 'yolo'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith('txt'):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Json to root of training_data
            try:
                source = "json_data"
                destination = self.json_label_path_data + "/" + 'json'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith('json'):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Transferring labels and json for ground truth
            try:
                def txt_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.txt'

                def json_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.json'

                filenames = next(os.walk('groundtruth_testing_data/data'), (None, None, []))[2]
                print(filenames, len(filenames))
                filenames_txt = list(map(txt_new, filenames))
                print(filenames_txt, len(filenames_txt))
                filenames_json = list(map(json_new, filenames))
                print(filenames_json, len(filenames_json))

                for fname_txt in filenames_txt:
                    src_path = os.path.join(self.json_label_path_data + '/yolo/', fname_txt)
                    dst_path = os.path.join('groundtruth_testing_data/data/', fname_txt)
                    shutil.move(src_path, dst_path)

                for fname_json in filenames_json:
                    src_path = os.path.join(self.json_label_path_data + '/json/', fname_json)
                    dst_path = os.path.join('groundtruth_testing_data/data/', fname_json)
                    shutil.move(src_path, dst_path)

                print('Generated Ground truth Data')
                self.custom_logger.info('Generated Ground truth Data')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            print(self.msg)
            self.custom_logger.info(self.msg)

            try:
                for root, dirs, files in os.walk(self.original_data_path):
                    for file in files:
                        if file.endswith(".names"):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join(self.json_label_path_data, file)
                            shutil.move(src_path, dst_path)
                print('Labels file moved to operations folder')
                self.custom_logger.info('Labels file moved to operations folder')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

        # ValidationPipeline raw data operations
        else:
            try:
                img_ground_truth_data = []
                txt_ground_truth_data = []
                for path_val_img, sub_dirs_val_img, files_val_img in os.walk(self.original_data_path):
                    for name in files_val_img:
                        if name.endswith(('.jpg', '.png')):
                            img_data_val = os.path.join(path_val_img, name)
                            img_ground_truth_data.append(img_data_val)
                        else:
                            pass

                for fname_image in img_ground_truth_data:
                    shutil.move(fname_image, 'groundtruth_testing_data/data/original images')

                for path_val_txt, sub_dirs_val_txt, files_val_txt in os.walk(self.original_data_path):
                    for name in files_val_txt:
                        if name.endswith('.txt'):
                            label_data_val = os.path.join(path_val_txt, name)
                            print(label_data_val)
                            txt_ground_truth_data.append(label_data_val)
                        else:
                            pass
                for fname_txt in txt_ground_truth_data:
                    shutil.move(fname_txt, 'groundtruth_testing_data/data/original Annotation')

                #print("Generated Validation Pipeline Data")
                self.custom_logger.info("Generated Validation Pipeline Data")
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Transferring json for Validation
            try:
                def json_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.json'

                filenames = next(os.walk('json_data'), (None, None, []))[2]
                filenames_json = list(map(json_new, filenames))

                for fname_json in filenames_json:
                    src_path = os.path.join('json_data', fname_json)
                    dst_path = os.path.join('groundtruth_testing_data/data/original Json', fname_json)
                    shutil.move(src_path, dst_path)

                #print('Generated Validation Pipeline Data')
                self.custom_logger.info('Generated Validation Pipeline Data')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            print(self.msg)
            self.custom_logger.info(self.msg)

            try:
                for root, dirs, files in os.walk(self.original_data_path):
                    for file in files:
                        if file.endswith(".names"):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join('groundtruth_testing_data/data/', file)
                            shutil.move(src_path, dst_path)
                print('Labels file moved to operations folder')
                self.custom_logger.info('Labels file moved to operations folder')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

        return self.msg

    def getFiles(path,filter="('.jpg', '.png')"):
        imagesList =[]
        absPath = os.path.join( os.getcwd() + path)
        for path_val, sub_dirs_val, files_val in os.walk(absPath):
            for name in files_val:
                if name.endswith(filter):    
                    #final_name = os.path.splitext(name)[0]
                    final_name = path_val+ "/"+name
                    imagesList.append(final_name)
        return imagesList

    
    def getFilesWithFolderName(self,path,extension_type):
        folderInfoList = []        
       
        for path_val, sub_dirs_val, files_val in os.walk(path):   
            imageList =[]
            for name in files_val:
                if name.endswith(extension_type):
                    imageList.append(path_val+ "/"+name)
                    
           
            if(len(imageList)  > 0):
                folder = path_val[len(path):len(path_val)]
                parent = os.path.dirname(os.path.dirname(folder)).replace('\\', '')
                folderInfoList.append({'parentFolder':parent,'folderName':path_val[len(path):len(path_val)],'files_list':imageList})
                self.logger.info('Considered '+str(len(imageList)) + ' from ' + path_val[len(path):len(path_val)])

        #remove list from root folder
       
        return folderInfoList
        
class DataLoop:
    
    def __init__(self, dataloop_config_dict, custom_logger, pipeline_type):
        self.custom_logger = custom_logger
        self.dataloop_config_dict = dataloop_config_dict
        self.pipeline_type = pipeline_type
        self.project_name_data = self.dataloop_config_dict.get('dl_project_name').strip()
        self.dataset_name_data = self.dataloop_config_dict.get('dl_dataset_name').strip()
        self.img_path_data = self.dataloop_config_dict.get('dl_img_path')
        self.json_label_path_data = self.dataloop_config_dict.get('dl_json_label_path')
        self.dl_filter = self.dataloop_config_dict.get('dl_filter')
        self.ground_truth_percentage = self.dataloop_config_dict.get('groundtruth_percentage')
        self.project = None
        self.dataset = None
        self.img_path = None
        self.json_label_path = None
        self.msg = None
        self.tag_list = None
        self.data_type = None
        if self.pipeline_type == const.TRAINING_PIPELINE:
            if os.path.exists("training_data"):
                shutil.rmtree("training_data")
            else:
                pass
            os.mkdir('training_data')
        else:
            pass
        # Refreshing Ground truth dataset
        if os.path.exists("groundtruth_testing_data"):
            shutil.rmtree("groundtruth_testing_data")
        else:
            pass
        os.mkdir('groundtruth_testing_data')
        os.mkdir('groundtruth_testing_data/data')

    def raw_data_operation(self):
        import dtlpy as dl
        try:
            if dl.token_expired():
                dl.login()
                # dl.login_m2m(email=self.dl_email, password=self.dl_password)
            print("Starting Download Activity from DataLoop to local folder")
            self.custom_logger.info("Starting Download Activity from DataLoop to local folder")
            # get project
            self.project = dl.projects.get(project_name=self.project_name_data)

            # get dataset
            self.dataset = self.project.datasets.get(dataset_name=self.dataset_name_data)

            # Entire dataset
            if self.dl_filter == "Dataset":
                # Download items
                filters = dl.Filters()
                # # Filter only annotated items + Approved
                filters.add(field='annotated', values=True)
                filters.add(field='metadata.system.taskStatusLog.status.status', values="approved")

                self.img_path = self.img_path_data
                self.json_label_path = self.json_label_path_data
                self.dataset.items.download(local_path=self.img_path, filters=filters)
                print("Download Done for Images")
                self.custom_logger.info("Download Done for Images")

                # Download labels
                converter = dl.Converter()
                converter.convert_dataset(dataset=self.dataset,
                                          to_format='yolo',
                                          local_path=self.json_label_path,
                                          filters=filters)
                print("Download Done for Annotation and metadata files")
                self.custom_logger.info("Download Done for Annotation and metadata files")

                self.msg = "Entire DatasetData Downloaded from data loop for training model in custom vision"

            # Incremental data download based on meta data properties
            elif self.dl_filter == "default":
                filters = dl.Filters()
                # # Filter only annotated and incremental items + Approved
                filters.add(field='annotated', values=True)
                filters.add(field='metadata.user.Downloaded', values="False")
                # filters.add(field='metadata.system.taskStatusLog.status.status', values="approved")

                self.img_path = self.img_path_data
                self.json_label_path = self.json_label_path_data
                self.dataset.items.download(local_path=self.img_path, filters=filters)
                print("Download Done for Images")
                self.custom_logger.info("Download Done for Images")
                # Download labels
                converter = dl.Converter()
                converter.convert_dataset(dataset=self.dataset,
                                          to_format='yolo',
                                          local_path=self.json_label_path,
                                          filters=filters)
                print("Download Done for Annotation and metadata files")
                self.custom_logger.info("Download Done for Annotation and metadata files")
                self.msg = "Incremental Data Downloaded from data loop for training model in custom vision"

            # Folder level download
            else:
                # Filter
                # # Filter only annotated and folder level + Approved
                filters = dl.Filters()
                filters.add(field='dir', values=self.dl_filter, operator=dl.FILTERS_OPERATIONS_EQUAL)
                filters.add(field='annotated', values=True)
                filters.add(field='metadata.system.taskStatusLog.status.status', values="approved")

                # Download items
                self.img_path = self.img_path_data
                self.json_label_path = self.json_label_path_data
                self.dataset.items.download(local_path=self.img_path, filters=filters)
                print("Download Done for Images")
                self.custom_logger.info("Download Done for Images")

                # Download labels
                converter = dl.Converter()
                converter.convert_dataset(dataset=self.dataset,
                                          to_format='yolo',
                                          local_path=self.json_label_path,
                                          filters=filters)
                print("Download Done for Annotation and metadata files")
                self.custom_logger.info("Download Done for Annotation and metadata files")

                self.msg = "Specified Folder Downloaded from data loop for training model in custom vision"

        except Exception as e:
            self.msg = "Could Not download Data"
            print("Could Not download Data from DataLoop")
            self.custom_logger.info("Could Not download Data from DataLoop")

        # Analysis of Raw data for Ground truth type selection
        try:
            categories = {'SO': [], 'MO': [], 'MOC': []}
            path = self.json_label_path_data
            self.data_type = identify_category(path, categories)
            # print(self.data_type)
            # print(self.data_type['SO'])
            # print(self.data_type['MO'])
            # print(self.data_type['MOC'])
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # Check whether MO,SO or MOC and decide Ground truth strategy

        # MOC GT Strategy
        if len(self.data_type['MOC']) > 0:
            # Transferring images for ground truth for MOC
            try:
                moc_ground_truth_data = []
                for path_val, sub_dirs_val, files_val in os.walk('training_data/Images'):
                    for name in files_val:
                        if name.endswith(('.jpg', '.png')):
                            meta_data_val = os.path.join(path_val, name)
                            moc_ground_truth_data.append(meta_data_val)
                        else:
                            pass
                moc_gt_data = len(moc_ground_truth_data)
                moc_gt_data_count = round(moc_gt_data * self.ground_truth_percentage)
                file_name = random.sample(moc_ground_truth_data, moc_gt_data_count)
                print(moc_gt_data, moc_gt_data_count, moc_ground_truth_data)
                for fname_image in file_name:
                    shutil.move(fname_image, 'groundtruth_testing_data/data')
                print("Generated ground truth Data class vise")
                self.custom_logger.info("Generated ground truth Data class vise")
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

        # MO SO GT Strategy
        else:
            # Transferring images for ground truth for SO and MO
            try:
                a = [os.path.join(path) for path, subdirs, files in os.walk('training_data/Images') for name in files]
                a = set(a)
                data_description = dict();
                for i in a:
                    _, _, files = next(os.walk(i))
                    file_count = len(files)
                    i = i.replace('\\', '/')
                    data_description[i] = round(file_count * self.ground_truth_percentage)
                print(data_description)
                self.custom_logger.info(data_description)
                for key, val in data_description.items():
                    dirpath = key
                    file_name = random.sample(os.listdir(dirpath), val)
                    for fname_txt in file_name:
                        src_path = os.path.join(dirpath, fname_txt)
                        shutil.move(src_path, 'groundtruth_testing_data/data')
                # print("Generated ground truth Data from downloaded DataLoop")
            except Exception as e:
                print("Could Not generate ground truth Data from downloaded DataLoop")
                self.custom_logger.exception("Could Not generate ground truth Data from downloaded DataLoop")

        # Images to root
        try:
            source = self.img_path_data + "/" + 'items' + "/"
            destination = self.img_path_data + "/" + 'items'
            for (root, dirs, files) in os.walk(source, topdown=True):
                for file in files:
                    path = os.path.join(root, file)
                    shutil.move(path, destination)
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # Labels to root
        try:
            source = self.json_label_path_data + "/" + 'yolo' + "/"
            destination = self.json_label_path_data + "/" + 'yolo'
            for (root, dirs, files) in os.walk(source, topdown=True):
                for file in files:
                    path = os.path.join(root, file)
                    shutil.move(path, destination)
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # Json to root
        try:
            source = self.json_label_path_data + "/" + 'json' + "/"
            destination = self.json_label_path_data + "/" + 'json'
            for (root, dirs, files) in os.walk(source, topdown=True):
                for file in files:
                    path = os.path.join(root, file)
                    shutil.move(path, destination)
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # Transferring labels and json for ground truth
        try:
            def txt_new(file_name_val):
                file_name_val = os.path.splitext(file_name_val)[0]
                return file_name_val + '.txt'

            def json_new(file_name_val):
                file_name_val = os.path.splitext(file_name_val)[0]
                return file_name_val + '.json'

            filenames = next(os.walk('groundtruth_testing_data/data'), (None, None, []))[2]
            # print(filenames)
            filenames_txt = list(map(txt_new, filenames))
            # print(filenames_txt)
            filenames_json = list(map(json_new, filenames))
            # print(filenames_json)

            for fname_txt in filenames_txt:
                src_path = os.path.join(self.json_label_path_data + '/yolo/', fname_txt)
                shutil.move(src_path, 'groundtruth_testing_data/data')

            for fname_json in filenames_json:
                src_path = os.path.join(self.json_label_path_data + '/json/', fname_json)
                shutil.move(src_path, 'groundtruth_testing_data/data')

            print('Generated Ground truth Data From Data downloaded from Dataloop')
            self.custom_logger.info('Generated Ground truth Data From Data downloaded from Dataloop')
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)

        # Updating Downloaded Metadata for training Images in Dataloop
        try:
            training_data = [img_file for img_file in os.listdir(self.img_path_data + "/" + 'items') if
                             not img_file.endswith(('.txt', '.json'))]
            pages = self.dataset.items.list()
            for page in pages:
                for item in page:
                    if item.name in training_data:
                        item.metadata['user']['Downloaded'] = 'True'
                        item.update()
                    else:
                        pass
            print('Successfully Updated Downloaded Metadata for Images in Dataloop')
            self.custom_logger.info('Successfully Updated Downloaded Metadata for Images in Dataloop')
        except Exception as e:
            print('Failed to Update Downloaded Metadata for Images in Dataloop')
            self.custom_logger.exception('Failed to Update Downloaded Metadata for Images in Dataloop')

        # Updating Ground truth and Downloaded Metadata for Ground Truth Images in Dataloop
        try:
            groundtruth_data = [img_file for img_file in os.listdir('groundtruth_testing_data/data') if
                                not img_file.endswith(('.txt', '.json'))]
            pages_gt = self.dataset.items.list()
            for page_gt in pages_gt:
                for item in page_gt:
                    if item.name in groundtruth_data:
                        item.metadata['user']['Groundtruth'] = 'True'
                        item.metadata['user']['Downloaded'] = 'True'
                        item.update()
                    else:
                        pass
            print('Successfully Updated Groundtruth Metadata for Images in Dataloop')
            self.custom_logger.info('Successfully Updated Groundtruth Metadata for Images in Dataloop')
        except Exception as e:
            print('Failed to Update Groundtruth Metadata for Images in Dataloop')
            self.custom_logger.exception('Failed to Update Groundtruth Metadata for Images in Dataloop')

        print(self.msg)
        self.custom_logger.info(self.msg)
        dl.logout()
        return self.msg


class S3:
    
    def __init__(self, aws_config_dict, custom_logger, pipeline_type):
        self.custom_logger = custom_logger
        self.aws_config_dict = aws_config_dict
        self.pipeline_type = pipeline_type
        self.aws_access_key_id = self.aws_config_dict.get('aws_access_key_id')
        self.aws_secret_access_key = self.aws_config_dict.get('aws_secret_access_key')
        self.bucket_name = self.aws_config_dict.get('bucket_name')
        self.s3_folder = self.aws_config_dict.get('s3_folder')
        self.local_dir = self.aws_config_dict.get('local_dir')
        self.img_path_data = self.aws_config_dict.get('raw_img_path')
        self.json_label_path_data = self.aws_config_dict.get('raw_json_label_path')
        self.ground_truth_percentage = self.aws_config_dict.get('ground_truth_percentage')
        self.bucket_val = None
        self.msg = None
        self.data_type = None
        if self.pipeline_type == const.TRAINING_PIPELINE:
            if os.path.exists("training_data"):
                shutil.rmtree("training_data")
            else:
                pass
            os.mkdir('training_data')
            os.mkdir('training_data/Images/')
            os.mkdir('training_data/Images/items')
            os.mkdir('training_data/labels/')
            os.mkdir('training_data/labels/yolo')
            os.mkdir('training_data/labels/json')
        else:
            pass
        if os.path.exists("json_data"):
            shutil.rmtree("json_data")
        else:
            pass
        os.mkdir('json_data')
        # Refreshing Ground truth dataset
        if os.path.exists("groundtruth_testing_data"):
            shutil.rmtree("groundtruth_testing_data")
        else:
            pass
        os.mkdir('groundtruth_testing_data')
        os.mkdir('groundtruth_testing_data/data')
        if self.pipeline_type == const.TRAINING_PIPELINE:
            pass
        else:
            os.mkdir('groundtruth_testing_data/data/original images')
            os.mkdir('groundtruth_testing_data/data/original Annotation')
            os.mkdir('groundtruth_testing_data/data/original Json')
            os.mkdir('groundtruth_testing_data/data/Predicted Annotation')

    def raw_data_operation(self):
        """
        Download the contents of a folder directory
        Args:
            bucket name: the name of the s3 bucket
            s3 folder: the folder path in the s3 bucket
            local dir: a relative or absolute directory path in the local file system
        """
        try:
            import boto3
            s3 = boto3.resource('s3', aws_access_key_id=self.aws_access_key_id,
                                aws_secret_access_key=self.aws_secret_access_key)

            self.bucket_val = s3.Bucket(self.bucket_name)
            for obj in self.bucket_val.objects.filter(Prefix=self.s3_folder):
                target = obj.key if self.local_dir is None else os.path.join(self.local_dir,
                                                                             os.path.relpath(obj.key, self.s3_folder))
                if not os.path.exists(os.path.dirname(target)):
                    os.makedirs(os.path.dirname(target))
                if obj.key[-1] == '/':
                    continue
                self.bucket_val.download_file(obj.key, target)

            self.msg = "Download of Data Completed"
        except Exception as e:
            self.msg = e

        # Generating json meta data JSON files for raw data from folder structure
        try:
            for path_val, sub_dirs_val, files_val in os.walk(self.local_dir):
                for name in files_val:
                    if name.endswith('.txt'):
                        meta_data = path_val.replace("\\", ",")
                        meta_data_dict = dict((x.strip(), y.strip())
                                              for x, y in (element.split('-')
                                                           for element in meta_data.split(',')))
                        final_name = os.path.splitext(name)[0]
                        create_json("json_data", final_name, meta_data_dict)
                    else:
                        pass
        except Exception as e:
            print(e)
            self.custom_logger.exception(e)
        # TrainingPipeline raw data operations
        if self.pipeline_type == const.TRAINING_PIPELINE:
            # Analysis of Raw data for Ground truth type selection
            try:
                categories = {'SO': [], 'MO': [], 'MOC': []}
                path = self.local_dir
                self.data_type = identify_category(path, categories)
                # print(self.data_type)
                # print(self.data_type['SO'])
                # print(self.data_type['MO'])
                # print(self.data_type['MOC'])
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Check whether MO,SO or MOC and decide Ground truth strategy
            # MOC GT Strategy
            if len(self.data_type['MOC']) > 0:
                # Transferring images for ground truth for MOC
                try:
                    moc_ground_truth_data = []
                    for path_val, sub_dirs_val, files_val in os.walk(self.local_dir):
                        for name in files_val:
                            if name.endswith(('.jpg', '.png')):
                                meta_data_val = os.path.join(path_val, name)
                                moc_ground_truth_data.append(meta_data_val)
                            else:
                                pass
                    moc_gt_data = len(moc_ground_truth_data)
                    moc_gt_data_count = round(moc_gt_data * self.ground_truth_percentage)
                    file_name = random.sample(moc_ground_truth_data, moc_gt_data_count)
                    print(moc_gt_data, moc_gt_data_count, moc_ground_truth_data)
                    for fname_image in file_name:
                        shutil.move(fname_image, 'groundtruth_testing_data/data')
                    print("Generated ground truth Data class vise")
                    self.custom_logger.info("Generated ground truth Data class vise")
                except Exception as e:
                    print(e)
                    self.custom_logger.exception(e)

            # MO SO GT Strategy
            else:
                # Transferring images for ground truth for SO and MO
                try:
                    gt_data = [os.path.join(path) for path, subdirs, files in
                               os.walk(self.local_dir) for name in files]
                    gt_data = set(gt_data)
                    data_description = dict();
                    for i in gt_data:
                        _, _, files = next(os.walk(i))
                        files_final = [i for i in files if i.endswith(('.jpg', '.png'))]
                        file_count = len(files_final)
                        i = i.replace('\\', '/')
                        print(file_count)
                        data_description[i] = round(file_count * self.ground_truth_percentage)
                    print(data_description)
                    for key, val in data_description.items():
                        dirpath = key
                        _, _, files_val = next(os.walk(dirpath))
                        files_final = [i for i in files_val if i.endswith(('.jpg', '.png'))]
                        file_name = random.sample(files_final, val)
                        for fname_image in file_name:
                            src_path = os.path.join(dirpath, fname_image)
                            shutil.move(src_path, 'groundtruth_testing_data/data')
                    print("Generated ground truth Data class vise")
                    self.custom_logger.info("Generated ground truth Data class vise")
                except Exception as e:
                    print("Could Not generate ground truth Data class vise")
                    self.custom_logger.exception("Could Not generate ground truth Data class vise")

            # Images to root of training_data
            try:
                source = self.local_dir
                destination = self.img_path_data + "/" + 'items'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith(('jpg', 'png')):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Labels to root of training_data
            try:
                source = self.local_dir
                destination = self.json_label_path_data + "/" + 'yolo'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith('txt'):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Json to root of training_data
            try:
                source = "json_data"
                destination = self.json_label_path_data + "/" + 'json'
                for (root, dirs, files) in os.walk(source, topdown=True):
                    for file in files:
                        if file.endswith('json'):
                            path = os.path.join(root, file)
                            shutil.move(path, destination)
                        else:
                            pass
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Transferring labels and json for ground truth
            try:
                def txt_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.txt'

                def json_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.json'

                filenames = next(os.walk('groundtruth_testing_data/data'), (None, None, []))[2]
                print(filenames, len(filenames))
                filenames_txt = list(map(txt_new, filenames))
                print(filenames_txt, len(filenames_txt))
                filenames_json = list(map(json_new, filenames))
                print(filenames_json, len(filenames_json))

                for fname_txt in filenames_txt:
                    src_path = os.path.join(self.json_label_path_data + '/yolo/', fname_txt)
                    dst_path = os.path.join('groundtruth_testing_data/data/', fname_txt)
                    shutil.move(src_path, dst_path)

                for fname_json in filenames_json:
                    src_path = os.path.join(self.json_label_path_data + '/json/', fname_json)
                    dst_path = os.path.join('groundtruth_testing_data/data/', fname_json)
                    shutil.move(src_path, dst_path)

                print('Generated Ground truth Data')
                self.custom_logger.info('Generated Ground truth Data')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            print(self.msg)
            self.custom_logger.info(self.msg)

            try:
                for root, dirs, files in os.walk(self.local_dir):
                    for file in files:
                        if file.endswith(".names"):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join(self.json_label_path_data, file)
                            shutil.move(src_path, dst_path)
                print('Labels file moved to operations folder')
                self.custom_logger.info('Labels file moved to operations folder')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

        # ValidationPipeline raw data operations
        else:
            try:
                img_ground_truth_data = []
                txt_ground_truth_data = []
                for path_val_img, sub_dirs_val_img, files_val_img in os.walk(self.original_data_path):
                    for name in files_val_img:
                        if name.endswith(('.jpg', '.png')):
                            img_data_val = os.path.join(path_val_img, name)
                            img_ground_truth_data.append(img_data_val)
                        else:
                            pass

                for fname_image in img_ground_truth_data:
                    shutil.move(fname_image, 'groundtruth_testing_data/data/original images')

                for path_val_txt, sub_dirs_val_txt, files_val_txt in os.walk(self.original_data_path):
                    for name in files_val_txt:
                        if name.endswith('.txt'):
                            label_data_val = os.path.join(path_val_txt, name)
                            print(label_data_val)
                            txt_ground_truth_data.append(label_data_val)
                        else:
                            pass
                for fname_txt in txt_ground_truth_data:
                    shutil.move(fname_txt, 'groundtruth_testing_data/data/original Annotation')
                #print("Generated Validation Pipeline Data")
                self.custom_logger.info("Generated Validation Pipeline Data")
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            # Transferring json for Validation
            try:
                def json_new(file_name_val):
                    file_name_val = os.path.splitext(file_name_val)[0]
                    return file_name_val + '.json'

                filenames = next(os.walk('json_data'), (None, None, []))[2]
                filenames_json = list(map(json_new, filenames))

                for fname_json in filenames_json:
                    src_path = os.path.join('json_data', fname_json)
                    dst_path = os.path.join('groundtruth_testing_data/data/original Json', fname_json)
                    shutil.move(src_path, dst_path)

                #print('Generated Validation Pipeline Data')
                self.custom_logger.info('Generated Validation Pipeline Data')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

            print(self.msg)
            self.custom_logger.info(self.msg)

            try:
                for root, dirs, files in os.walk(self.original_data_path):
                    for file in files:
                        if file.endswith(".names"):
                            src_path = os.path.join(root, file)
                            dst_path = os.path.join('groundtruth_testing_data/data/', file)
                            shutil.move(src_path, dst_path)
                print('Labels file moved to operations folder')
                self.custom_logger.info('Labels file moved to operations folder')
            except Exception as e:
                print(e)
                self.custom_logger.exception(e)

        return self.msg
