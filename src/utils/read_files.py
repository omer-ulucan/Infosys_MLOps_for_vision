'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import json
import os


class File(object):
    """[summary]

    Args:
        object ([type]): [description]
    """
    def __init__(self, filepath: str) -> None:
        if not filepath.endswith('.json'):
            raise ValueError("filepath must be an .json file:", filepath)

        self.filepath = filepath

    def read(self) -> dict:
        with open(self.filepath, "r") as f:
            data = json.load(f)

        return data


if __name__ == "__main__":

    this_dir = os.path.dirname(os.path.abspath(__file__))
    example_file = os.path.join(this_dir, 'example.json')

    annot_file = File(example_file)
    data = annot_file.read()
