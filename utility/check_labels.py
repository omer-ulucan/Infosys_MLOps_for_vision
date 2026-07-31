'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os


def check_class_count(annotation_file_path, labels_file_path):
    textFilePath = annotation_file_path
    labelFilePath = labels_file_path
    data = os.listdir(textFilePath)
    labelFile = [line.strip() for line in open(labelFilePath)]
    # print(labelFile)
    labelsCount = {}
    count = 0
    for file in data:
        if file.endswith('.txt'):
            with open(os.path.join(textFilePath, file), 'r') as annotationFile:
                labelData = annotationFile.readlines()
                for line in labelData:
                    label = int(line.split(' ')[0])
                    if label in labelsCount:
                        labelsCount[label] += 1
                    else:
                        labelsCount[label] = 1
    # print(labelsCount)
    class_labels_count = {}
    for count_key, count_value in labelsCount.items():
        count_key = labelFile[count_key]
        class_labels_count[count_key] = count_value
    return class_labels_count
