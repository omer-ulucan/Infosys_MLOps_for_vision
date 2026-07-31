'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os

def create_confusion_matrix(data_list):
    confusion_matrix = defaultdict(int)

    for item in data_list:
        ground_truth = item.get("GT", {}).get("class_name")
        prediction = item.get("predictions", {}).get("best_match").get("class_name")

        if ground_truth is not None and prediction is not None:
            confusion_matrix[(ground_truth, prediction)] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix,file_name, title='Confusion matrix', cmap=plt.cm.Blues):
    classes = set()
    for actual, predicted in confusion_matrix:
       
        classes.add(actual)
        classes.add(predicted)
    classes = sorted(list(classes))

    matrix = np.zeros((len(classes), len(classes)))
    for class_combo, no_of_pred in confusion_matrix.items():
        actual = class_combo[0]
        predicted = class_combo[1]
        matrix[classes.index(actual), classes.index(predicted)] = no_of_pred

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = matrix.max() / 2.
    for i, j in np.ndindex(matrix.shape):
        plt.text(j, i, format(matrix[i, j]),
                 ha="center", va="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(file_name)) #save the figure
    plt.close() #close the figure


