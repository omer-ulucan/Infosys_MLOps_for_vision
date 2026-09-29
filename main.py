'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Main Class for Pipeline Builder Factory pattern """

import os
import sys
from module.logger import BPLogger
from pipeline_manager import PipeLineManager
from config_manager import ConfigManager
from utility.json_ops import get_dict_from_json
import traceback
import utility.constants as const

class MLOPSPipeline:
    """ Main class which performs tasks like load config, create pipeline and run
    pipeline """

    def __init__(self, raw_data_source=None, pipeline_platform=None, gt_report=None, pipeline_type=None,
                 platform_config_path=None):
        # basic variables
        self.raw_data_source = raw_data_source
        self.pipeline_platform = pipeline_platform
        self.gt_report = gt_report
        self.pipeline_type = pipeline_type
        self.platform_config_path = platform_config_path
        self.custom_pipeline_yolo_version = None
        if self.raw_data_source is None and self.pipeline_platform is None and gt_report is None:
            primary_config = get_dict_from_json('framework_config/vision_sdk_primary_config.json')
            self.raw_data_source = primary_config['raw_data_source']
            self.pipeline_platform = primary_config['pipeline_platform']
            self.gt_report = primary_config['gt_report']
            self.pipeline_type = primary_config['pipeline_type']
            self.custom_pipeline_yolo_version = primary_config['custom_pipeline_yolo']
            self.platform_config_path = primary_config.get('platform_config')
            if self.pipeline_platform == "CustomYoloPipeline":
                self.pipeline_platform = self.pipeline_platform + self.custom_pipeline_yolo_version
            else:
                self.pipeline_platform = self.pipeline_platform
        else:
            pass
        # config manager instance for loading configs
        self.config_manager = ConfigManager(self.raw_data_source, self.gt_report, self.pipeline_platform,
                                            self.pipeline_type, platform_config_path=self.platform_config_path)
        # Default values
        self.general_config = None
        self.raw_data_config = None
        self.pipeline_config = None
        self.gt_report_config = None
        self.pipeline_type_config = None
        self.logger = None
        self.pipeline_builder = None
        self.logging = None
        self.iteration_status = None

    def load_config(self):
        """ Load all necessary config and perform precedence of params based on hierarchy"""
        try:
            if self.pipeline_type ==const.TRAINING_PIPELINE:
                # Loading all configs from file or Remote for TrainingPipeline
                self.general_config = self.config_manager.get_general_config()
                self.raw_data_config = self.config_manager.get_raw_data_config()
                self.pipeline_config = self.config_manager.get_platform_config()
                self.gt_report_config = self.config_manager.get_gt_report_config()
                self.pipeline_type_config = self.config_manager.get_pipeline_type_config()
            elif self.pipeline_type ==const.VALIDATION_PIPELINE:
                # Loading all configs from file or Remote ValidationPipeline
                self.general_config = self.config_manager.get_general_config()
                print("general_config is loaded")
                self.raw_data_config = self.config_manager.get_raw_data_config()
                print("raw_data_config is loaded")
                self.gt_report_config = self.config_manager.get_gt_report_config()
                print("gt_report_config is loaded")
                self.pipeline_config = self.config_manager.get_platform_config()
                print("pipeline_config is loaded")
                # self.pipeline_type_config = self.config_manager.get_pipeline_type_config()
            elif self.pipeline_type ==const.VALIDATION_BY_EMBEDDINGS_PIPELINE:
                # Loading embeddings pipeline config
                self.general_config = self.config_manager.get_general_config()
                print("general_config is loaded")
                self.raw_data_config = self.config_manager.get_embeddingsval_config()
                print("raw_data_config is loaded")
                self.pipeline_config = self.config_manager.get_embeddingsval_config()
                print("pipeline_config is loaded")
            elif self.pipeline_type ==const.FEATURE_EXTRACTION_PIPELINE:
                # Loading embeddings pipeline config
                self.general_config = self.config_manager.get_general_config()
                print("general_config is loaded")
                self.raw_data_config = self.config_manager.get_embeddings_config()
                print("raw_data_config is loaded")
                self.pipeline_config = self.config_manager.get_embeddings_config()
                print("pipeline_config is loaded")
            elif self.pipeline_type ==const.AUGMENTATION_PIPELINE:
                # Loading embeddings pipeline config
                self.general_config = self.config_manager.get_general_config()
                print("general_config is loaded")
                self.raw_data_config = self.config_manager.get_augmentations_config()
                print("raw_data_config is loaded")
                self.pipeline_config = self.config_manager.get_augmentations_config()
                print("pipeline_config is loaded")
            else:
                # Loading all config files for FeatureExtractionPipeline
                self.general_config = self.config_manager.get_general_config()
                self.raw_data_config = self.config_manager.get_raw_data_config()
                self.pipeline_config = self.config_manager.get_platform_config()
                self.pipeline_type_config = self.config_manager.get_pipeline_type_config()
        except Exception as e:
            traceback.print_exc()

    # Creating common components - Logger
    def create_components(self):
        """ Create common components like logger, exception handler etc"""
        try:
            # Bp logger instance for common components
            self.logging = BPLogger(self.general_config)
            self.logger = self.logging.log_config_loader()
        except Exception as e:
            traceback.print_exc()

    # Build Pipeline using Pipeline manager and run the pipeline (Training Pipeline or Validation Pipeline)
    def create_and_run_pipeline(self):
        """ Create pipeline using associated class and run the created pipeline """
        try:
            # Training or Validation Pipeline logic
            self.iteration_status = os.path.exists("training_data")
            self.pipeline_builder = PipeLineManager(raw_data_source=self.raw_data_source,
                                                    raw_data_config=self.raw_data_config,
                                                    pipeline_type=self.pipeline_type,
                                                    pipeline_type_config=self.pipeline_type_config,
                                                    logger=self.logger,
                                                    gt_report_config=self.gt_report_config,
                                                    custom_pipeline_yolo_version=self.custom_pipeline_yolo_version,
                                                    pipeline_config=self.pipeline_config,
                                                    iteration_status=self.iteration_status,
                                                    pipeline_platform=self.pipeline_platform)
            self.pipeline_builder.raw_data_operation()
            self.pipeline_builder.build_pipeline()
        except Exception as e:
            traceback.print_exc()


if __name__ == "__main__":
    # raw_data_source - DataLoop, Local, S3
    # pipeline_platform - AzureCustomVision, AICloud
    params_list = len(sys.argv)
    # print(params_list)
    if params_list <= 1:
        result = MLOPSPipeline()
    else:
        result = MLOPSPipeline(raw_data_source=sys.argv[1],
                               pipeline_platform=sys.argv[2],
                               gt_report=sys.argv[3],
                               pipeline_type=sys.argv[4])
    print("loading config ..")
    result.load_config()
    print("config loaded..")
    result.create_components()
    print("component created..")
    result.create_and_run_pipeline()
    print("pipeline is running..")
