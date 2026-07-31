'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

""" Module gets all configurations from file on input for Download and platform and
ground truth reports and pipeline type returns dictionary object to be used in other modules """
import json
from utility.json_ops import get_dict_from_json

# DataLoop Source
class DataLoop:
    """ Loads Config file for Dataloop and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_general_config(self):
        """ Func for general config """
        try:
            self.path = "framework_config/general_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)

    def load_raw_data_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/dataloop_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# Local folder Source

class Local:
    """ Loads Config file for Local and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_general_config(self):
        """ Func for general config """
        try:
            self.path = "framework_config/general_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)

    def load_raw_data_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/local_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# AWS S3 Source
class S3:
    """ Loads Config file for S3 and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_general_config(self):
        """ Func for general config """
        try:
            self.path = "framework_config/general_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)

    def load_raw_data_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/aws_s3.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# ACV Platform
class AzureCustomVision:
    """ Loads Config file for ACV and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/azure_custom_vision_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# AICloud Platform
class AICloud:
    """ Loads Config file for AICloud and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/ai_cloud_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# Custom Yolo-V7 Pipeline Platform
class CustomYoloPipelineYoloV7:
    """ Loads Config file for Custom Yolo-V7 Pipeline and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/yolov7-custom_training_pipeline.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# Custom Yolo-V4 Pipeline Platform
class CustomYoloPipelineYoloV4:
    """ Loads Config file for Custom Yolo-V6 Pipeline and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline raw data operation config """
        try:
            self.path = "framework_config/yolov4-custom_training_pipeline.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# Custom Yolo-V8 Pipeline Platform
class CustomYoloPipelineYoloV8:
    """ Loads Config file for Custom Yolo-V8 Pipeline and returns Dictionary """

    def __init__(self, override_path=None):
        """ Initialising variables values """
        self.path = override_path
        self.config_dict = None

    def load_platform_config(self):
        """ Func for pipeline raw data operation config """
        try:
            if not self.path:
                self.path = "framework_config/yolov8-custom_training_pipeline.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# gt_reports Config Loading
class GTReport:
    """ Loads Config file for gt_reports and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_gt_report_config(self):
        """ Func for pipeline gt report config """
        try:
            self.path = "framework_config/gt_categories.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)
    
    def load_embdvalidation__config(self):
        """ Func for embeddings pipeline config """
        try:
            self.path = "framework_config/validation_by_embeddings.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# IVA Validation Pipeline Config Loading
class IVAValidationPipeline:
    """ Loads Config file for Validation Training and returns Dictionary """

    def __init__(self, override_path=None):
        """ Initialising variables values """
        self.path = override_path
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline gt report config """
        try:
            if not self.path:
                self.path = "framework_config/iva_validation_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# ACV Validation Pipeline Config Loading
class ACVValidationPipeline:
    """ Loads Config file for Validation Training and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline gt report config """
        try:
            self.path = "framework_config/acv_validation_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# AWSValidation Pipeline Config Loading
class AWSValidationPipeline:
    """ Loads Config file for Validation Training and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    

    def load_platform_config(self):
        """ Func for pipeline gt report config """
        try:
            self.path = "framework_config/aws_validation_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


# Training Pipeline Config Loading
class TrainingPipeline:
    """ Loads Config file for Training Pipeline and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    @staticmethod
    def load_pipeline_type_config():
        try:
            pass
        except Exception as e:
            print(e)


# Training Pipeline Config Loading
class ValidationPipeline:
    """ Loads Config file for Training Pipeline and returns Dictionary """

    def __init__(self):
        """ Initialising variables values """
        self.path = None
        self.config_dict = None

    @staticmethod
    def load_pipeline_type_config():
        try:
            pass
        except Exception as e:
            print(e)

class ValidationByEmbeddingsPipeline:
    """ Loads Config file for Validation Training and returns Dictionary """

    def __init__(self, override_path=None):
        """ Initialising variables values """
        self.path = override_path
        self.config_dict = None

    

    def load_embeddingsval_config(self):
        """ Func for pipeline gt report config """
        try:
            if not self.path:
                self.path = "framework_config/validation_by_embeddings.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)
class FeatureExtractionPipeline:
    """ Loads Config file for Validation Training and returns Dictionary """

    def __init__(self, override_path=None):
        """ Initialising variables values """
        self.path = override_path
        self.config_dict = None
    def load_embeddings_config(self):
        """ Func for pipeline gt report config """
        try:
            if not self.path:
                self.path = "framework_config/feature_extraction.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)


class AugmentationPipeline:

    def __init__(self, override_path=None):
        """ Initialising variables values """
        self.path = override_path
        self.config_dict = None
    def load_augmentations_config(self):
        """ Func for pipeline gt report config """
        try:
            if not self.path:
                self.path = "framework_config/augmentations_config.json"
            self.config_dict = get_dict_from_json(self.path)
            return self.config_dict
        except Exception as e:
            print(e)
