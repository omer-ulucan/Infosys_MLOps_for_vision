'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

""" Custom Logger class for using in Design Framework """

# DEBUG: Detailed information, typically of interest only when diagnosing problems.

# INFO: Confirmation that things are working as expected.

# WARNING: An indication that something unexpected happened, or indicative of some problem in the near future (e.g.
# ‘disk space low’). The software is still working as expected.

# ERROR: Due to a more serious problem, the software has not been able to perform some function.

# CRITICAL: A serious error, indicating that the program itself may be unable to continue running.


import logging
from module.file_logger import log_in_file
from module.RDBMS_logger_handler import DBHandler


class BPLogger:
    """ Reads framework_config json file and sets the log levels and type based on it """

    def __init__(self, general_config_dict):
        """ Log level and type set to default values if there is no input from user in framework_config files """
        self.config_dict = general_config_dict
        self.logger = logging.getLogger("MLOPS_Logger")
        self.log_type = None
        self.file_handler = None
        self.logging_db_type = None
        self.db_handler = None
        self.log_file = None
        self.path_name = None
        self.log_file_info = None
        self.log_level = None
        self.formatter = None
        self.logging_db_engine = None

    def log_config_loader(self):
        """ Set log level, type according to inputs from framework_config file """
        try:

            # Setting log level Number from framework_config file
            # * (20 = Developer mode = INFO) and (other, 30 = Production mode = Error)
            self.log_level = self.config_dict.get('logging_level')

            if self.log_level == 20:
                self.logger.setLevel(logging.INFO)
                self.formatter = logging.Formatter('%(name)s : %(asctime)s : %(levelname)s : %('
                                                   'levelno)s :  %(message)s')
            else:
                self.logger.setLevel(logging.ERROR)
                self.formatter = logging.Formatter('%(name)s : %(asctime)s : %(levelname)s : %('
                                                   'levelno)s :  %(message)s')

            # Setting log type from framework_config
            self.log_type = self.config_dict.get('logging_type')

            # Setting logging related db type and connection from framework_config
            self.logging_db_type = self.config_dict.get('logging_db_type')
            self.logging_db_engine = self.config_dict.get('logging_db_engine')

            # Log-type File related configuration
            if self.log_type == "File":
                self.log_file_info = log_in_file(self.config_dict)
                self.file_handler = logging.FileHandler(self.log_file_info)
                self.file_handler.setFormatter(self.formatter)
                self.logger.addHandler(self.file_handler)

            # Log-type DB related configuration

            elif self.log_type == "DB":
                if self.logging_db_type == "RDBMS":
                    self.db_handler = DBHandler(self.logging_db_engine)
                    self.db_handler.setFormatter(self.formatter)
                    self.logger.addHandler(self.db_handler)
                else:
                    pass

            # Log-type Console related configuration
            else:
                stream_handler = logging.StreamHandler()
                stream_handler.setFormatter(self.formatter)
                self.logger.addHandler(stream_handler)
            return self.logger

        except Exception as e:
            print(e)
