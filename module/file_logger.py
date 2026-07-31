'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

""" Module to handle logging in file and Log to default filename if no file name is provided """

import pathlib
import os
import time


def log_in_file(config_dict):
    """ Creates folder for logging file and logs to static file if no filename is provided """
    try:
        log_file_data = config_dict.get('logging_file')
        curr_time = time.strftime("%H_%M_%S", time.localtime())
        curr_time_final = str(curr_time)
        if log_file_data is not None:
            log_file_data_final = log_file_data.split('.')[0]
            log_file = f"Logs/{log_file_data_final}{curr_time_final}.log"
        else:
            log_file = f"Logs/test_log{curr_time_final}.log"
        # Creating path directory from framework_config file for log file location or creating default path of
        # Logs/filename
        path_name, _ = os.path.split(log_file)
        pathlib.Path(path_name).mkdir(parents=True, exist_ok=True)
        return log_file
    except Exception as e:
        print(e)
