'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Load and Build pipeline from pipeline builder main file as per user defined input params on pipeline platform """
import importlib
import json
import utility.constants as const
import traceback

class PipeLineManager:
    """ Manages config and building of pipeline based on user input  """

    def __init__(self, raw_data_source, raw_data_config, pipeline_type, pipeline_platform, logger, gt_report_config,
                 custom_pipeline_yolo_version=None, pipeline_config=None, iteration_status=None,
                 pipeline_type_config=None):
        self.raw_data_source = raw_data_source
        self.raw_data_config = raw_data_config
        self.pipeline_type = pipeline_type
        self.pipeline_type_config = pipeline_type_config
        self.logger = logger
        self.gt_report_config = gt_report_config
        self.pipeline_config = pipeline_config
        self.iteration_status = iteration_status
        self.pipeline_platform = pipeline_platform
        self.custom_pipeline_yolo_version = custom_pipeline_yolo_version
        self.raw_data_class = getattr(importlib.import_module("module.raw_training_data"),
                                      self.raw_data_source)
        if self.pipeline_type == const.TRAINING_PIPELINE:
            self.pipeline_class = getattr(importlib.import_module("module.model_training_pipeline"),
                                          self.pipeline_platform)
        elif self.pipeline_type == const.VALIDATION_BY_EMBEDDINGS_PIPELINE:
            self.pipeline_class = getattr(importlib.import_module("module.embeddings_pipeline"),
                                          self.pipeline_type)
            
        elif self.pipeline_type == const.FEATURE_EXTRACTION_PIPELINE:
            self.pipeline_class = getattr(importlib.import_module("module.embeddings_pipeline"),
                                          self.pipeline_type)
        elif self.pipeline_type == const.AUGMENTATION_PIPELINE:
            self.pipeline_class = getattr(importlib.import_module("module.augmentations_pipeline"),
                                          self.pipeline_type)
        else:
            self.pipeline_class = getattr(importlib.import_module("module.validation_pipeline"),
                                          self.pipeline_platform)
        self.data_result = None
        self.pipeline_result = None

    def raw_data_operation(self):
        """ Function that performs raw data operations """
        # Training Pipeline raw data operation
        if self.pipeline_type == const.TRAINING_PIPELINE:
            try:
                if self.iteration_status:
                    pass
                else:
                    self.data_result = self.raw_data_class(self.raw_data_config, self.logger, self.pipeline_type)
                    self.data_result.raw_data_operation()
            except Exception as e:
                #print(e)
                traceback.print_exc()
        # Validation Pipeline raw data operation
        else:
            try:
               self.data_result = self.raw_data_class(self.raw_data_config, self.logger, self.pipeline_type)
               self.data_result.raw_data_operation()
            except Exception as e:
                #print(e)
                traceback.print_exc()

    def build_pipeline(self):
        """ Function that creates pipeline """
        # Training Pipeline
        if self.pipeline_type == const.TRAINING_PIPELINE:
            try:
                self.pipeline_result = self.pipeline_class(self.pipeline_config, self.logger, self.gt_report_config,
                                                           self.custom_pipeline_yolo_version)
                if self.iteration_status:
                    self.pipeline_result.perform_pipeline_operation()
                    #self.pipeline_result.clean_resources()
                else:
                    self.pipeline_result.perform_augmentation()
                    self.pipeline_result.perform_pipeline_operation()
                    #self.pipeline_result.clean_resources()
            except Exception as e:
                # print(e)
                traceback.print_exc()
        elif self.pipeline_type == const.FEATURE_EXTRACTION_PIPELINE:
            try:
                self.logger.info("Feature extraction pipeline started")
                self.pipeline_obj = self.pipeline_class(self.logger,self.pipeline_config, self.gt_report_config) 
                self.pipeline_obj.extract_and_save_features()

            except Exception as e:
                self.logger.error(e)
                traceback.print_exc()
        elif self.pipeline_type == const.VALIDATION_BY_EMBEDDINGS_PIPELINE:
            try:
                self.logger.info("Validation Pipeline by Embeddings started")
                self.pipeline_obj = self.pipeline_class(self.logger,self.pipeline_config, self.gt_report_config) 
                
                #validation flow      
                self.logger.info("Extracting validation dataset images features")         
                self.pipeline_obj.extract_validation_dataset_features()

                self.logger.info("Loading template embeddings")
                embeddings = self.pipeline_obj.load_template_embeddings()
                if(len(embeddings)  > 0):

                    self.logger.info("Checking similarity scores")
                    self.pipeline_obj.check_similarity_score()    

                    self.logger.info("Generating validation report") 
                    self.pipeline_obj.generate_validation_report()  

                    self.logger.info("Validation pipeline execution is completed")  
                else:
                    self.logger.error("Validation pipeline execution is stopped as no template embeddings are found")  
            
            except Exception as e:
                #print(e)
                traceback.print_exc()

        elif self.pipeline_type == const.AUGMENTATION_PIPELINE:
            try:
                self.logger.info("Augmentations Pipeline Started")
                self.pipeline_obj = self.pipeline_class(self.logger,self.pipeline_config, self.gt_report_config) 
                
                #Augmentations flow      
                self.logger.info("Applying transformation to input image")         
                self.pipeline_obj.perform_augmentation()
 

                self.logger.info("Augmentation pipeline execution is completed")  
                
            except Exception as e:
                traceback.print_exc()

        # Validation Pipeline
        else:
            try:
                print("building pipeline..")
                self.pipeline_result = self.pipeline_class(self.pipeline_config, self.gt_report_config)
                print("pipeline_class initiated")
                # Dispatch to patch-based flow when patch_based_inference is enabled in the
                # platform config. run_validation() drives PDF -> images -> patches -> YOLO ->
                # stitch -> Excel reports (patch_outputs/, pdf_image_output_folder/,
                # validation_reports/). Otherwise fall back to the standard API-based flow.
                if getattr(self.pipeline_result, 'patch_based_inference_enabled', False) \
                        and getattr(self.pipeline_result, 'patch_based_enabled', False):
                    print("patch-based inference enabled — running run_validation()")
                    self.pipeline_result.run_validation(multi_object_mode=True, generate_reports=True)
                    print("patch validation complete")
                else:
                    self.pipeline_result.inference_images()
                    print("images inferenced")
                    self.pipeline_result.generate_summary_report()
                    print("generated summary report")
                #self.pipeline_result.clean_resources()
                print("resources cleaned")
            except Exception as e:
                #print(e)
                traceback.print_exc()
                print("Exception in build_pipeline : ",e)
