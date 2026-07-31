'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Config manager is used to select method(File or Remote) and produce configuration data """
import importlib


class ConfigManager:
    """ Loads Config file from File or REST API as per input and returns configuration dictionary """

    # Map pipeline_type class name -> which loader instance receives the platform_config override.
    # Values are attribute names on `self` after all class instantiations below.
    _PLATFORM_CONFIG_ROUTES = {
        "TrainingPipeline": "platform_config_class",
        "ValidationPipeline": "platform_config_class",
        "AugmentationPipeline": "get_augmentations_config_class",
        "FeatureExtractionPipeline": "get_embeddings_config_class",
        "ValidationByEmbeddingsPipeline": "get_embeddingsval_config_class",
    }

    def __init__(self, raw_data_config_file, gt_report_config_file, platform_config_file,
                 pipeline_type_config_file=None, platform_config_path=None):
        """ Initialising variables values """
        # Raw data Config
        self.raw_data_config_file = raw_data_config_file
        self.raw_data_config_obj = getattr(importlib.import_module("module.config_handler"), self.raw_data_config_file)

        # Raw platform Config
        self.platform_config_file = platform_config_file
        self.platform_config_obj = getattr(importlib.import_module("module.config_handler"), self.platform_config_file)

        # Raw Gt report Config
        self.gt_report_config_file = gt_report_config_file
        self.gt_report_config_obj = getattr(importlib.import_module("module.config_handler"),
                                            self.gt_report_config_file)
        # Pipeline type Config
        self.pipeline_type_config_file = pipeline_type_config_file
        self.pipeline_type_config_obj = getattr(importlib.import_module("module.config_handler"),
                                                self.pipeline_type_config_file)

        # embeddings Config
        self.embeddings_val_config_file = pipeline_type_config_file
        self.embeddings_config_file = pipeline_type_config_file
        self.augmentations_config_file = pipeline_type_config_file

        self.embeddings_val_config_obj = getattr(importlib.import_module("module.config_handler"),
                                                self.embeddings_val_config_file)
        self.embeddings_config_obj = getattr(importlib.import_module("module.config_handler"),
                                                self.embeddings_config_file)
        self.augmentations_config_obj = getattr(importlib.import_module("module.config_handler"),
                                                self.augmentations_config_file)

        # Determine which loader (if any) receives the platform_config override.
        route_target = self._PLATFORM_CONFIG_ROUTES.get(pipeline_type_config_file) if platform_config_path else None

        self.raw_data_config_class = self.raw_data_config_obj()
        self.platform_config_class = (self.platform_config_obj(override_path=platform_config_path)
                                      if route_target == "platform_config_class"
                                      else self.platform_config_obj())
        self.gt_report_config_class = self.gt_report_config_obj()
        self.pipeline_type_config_class = self.pipeline_type_config_obj()
        self.get_embeddingsval_config_class = (self.embeddings_val_config_obj(override_path=platform_config_path)
                                               if route_target == "get_embeddingsval_config_class"
                                               else self.embeddings_val_config_obj())
        self.get_embeddings_config_class = (self.embeddings_config_obj(override_path=platform_config_path)
                                            if route_target == "get_embeddings_config_class"
                                            else self.embeddings_config_obj())
        self.get_augmentations_config_class = (self.augmentations_config_obj(override_path=platform_config_path)
                                              if route_target == "get_augmentations_config_class"
                                              else self.augmentations_config_obj())
        self.config_data = None
        self.raw_data_config_data = None
        self.platform_config_data = None
        self.gt_report_config_data = None
        self.pipeline_type_config_data = None
        self.embeddings_pipeline_config_data = None
        self.augmentations_pipeline_config_data = None

    def get_general_config(self):
        """ Func for getting general config data """
        try:
            self.config_data = self.raw_data_config_class.load_general_config()
            return self.config_data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_raw_data_config(self):
        """ Func for getting raw data config data """
        try:
            self.raw_data_config_data = self.raw_data_config_class.load_raw_data_config()
            return self.raw_data_config_data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_platform_config(self):
        """ Func for getting pipeline structure and config data """
        try:
            self.platform_config_data = self.platform_config_class.load_platform_config()
            return self.platform_config_data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_gt_report_config(self):
        """ Func for getting ground truth config data """
        try:
            self.gt_report_config_data = self.gt_report_config_class.load_gt_report_config()
            return self.gt_report_config_data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_pipeline_type_config(self):
        """ Func for getting pipeline type config data """
        try:
            self.pipeline_type_config_data = self.pipeline_type_config_class.load_pipeline_type_config()
            return self.pipeline_type_config_data
        except Exception as e:
            print(e)
            traceback.print_exc()

    def get_embeddingsval_config(self):
        """ Func for getting embeddings validation config data """
        try:
            self.embeddings_pipeline_config_data = self.get_embeddingsval_config_class.load_embeddingsval_config()
            return self.embeddings_pipeline_config_data
        except Exception as e:
            print(e)

    def get_embeddings_config(self):
        """ Func for getting embeddings validation config data """
        try:
            self.embeddings_pipeline_config_data = self.get_embeddings_config_class.load_embeddings_config()
            return self.embeddings_pipeline_config_data
        except Exception as e:
            print(e)

    def get_augmentations_config(self):
        """ Func for getting embeddings validation config data """
        try:
            self.augmentations_pipeline_config_data = self.get_augmentations_config_class.load_augmentations_config()
            return self.augmentations_pipeline_config_data
        except Exception as e:
            print(e)
