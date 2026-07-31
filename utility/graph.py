'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
# importing the required module
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel(
    r'C:\Users\parthshailesh.c\PycharmProjects\Master Auto Annotation SDK_19_1_23-Flask App-Final\static\bangalore it data.xlsx')

plt.plot(df['Iteration No.'], df['Precision'], linestyle='--', marker='o')
plt.plot(df['Iteration No.'], df['Recall'], linestyle='--', marker='o')
plt.plot(df['Iteration No.'], df['mAP'], linestyle='--', marker='o')
# plt.bar(df['Iteration No.'], df['No. of Training Images'])
# naming the x axis
plt.xlabel('Iteration No.')
# plt.ylabel('No. of Training Images')
# plt.xticks([1, 2, 3, 4])
# giving a title to my graph
plt.title('Iteration No. vs Precision vs Recall vs mAP')

# function to show the plot
plt.show()
