'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import mlflow
import uuid
import os

uuid_custom = uuid.uuid1()
mlflow.set_experiment('Bangalore_Demo_Product_detection')
mlflow.log_param('Project Name', 'Bangalore_Demo_Product_detection')
mlflow.log_param('Details of Training images Part', '6282')
mlflow.log_param('Details of Augmentation', str([{"HorizontalFlip" : {"p": 1},"Rotate" : {"limit" : 30, "p" : 1},
  "GaussianBlur" : {"p" : 1}}]))
mlflow.log_param('Details of Special Augmentation', 'None')
mlflow.log_param('Project Id', '88e66223-a01b-484c-b16c-8080effac94d')
mlflow.log_param('Iteration Id', 'fa8c83fb-c28b-4a4a-9666-82b1a921c834')
mlflow.log_param('Custom Iteration Id', uuid_custom)
mlflow.log_param('Probability Threshold', 0.5)
mlflow.log_param('Overlap Threshold', 0.3)
mlflow.log_metric('Precision', 92.8)
mlflow.log_metric('Recall', 40.5)
mlflow.log_metric('MAP', 75.8)

os.system('mlflow ui')
