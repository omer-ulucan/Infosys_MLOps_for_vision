'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import os


def identify_category(path, categories):
    for path_val, sub_dirs_val, files_val in os.walk(path):
        for file in files_val:
            if file.endswith('.txt'):
                with open(path_val + "/" + file, "r") as fp:
                    data = fp.readlines()
                    indexes = set()
                    if len(data) == 1:
                        categories['SO'].append(file)
                    else:
                        for labelling in data:
                            indexes.add(labelling[0:2].strip())
                        if len(indexes) > 1:
                            categories['MOC'].append(file)
                        elif len(indexes) == 1:
                            categories['MO'].append(file)
    return categories

