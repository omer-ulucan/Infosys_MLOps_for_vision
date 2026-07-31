'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" Custom DBHandler handler class for logging logs into RDBMS DB """

from logging import Handler
from module.RDBMS_logger_crud import DbLoggerCrud
from module.RDBMS_logger_schema import DBLog


class DBHandler(Handler):
    """ Used for logging logs in RDBMS DB"""
    def __init__(self, connection):
        self.connection = connection
        super().__init__()

    def emit(self, record):
        message = self.format(record)
        try:
            new_log = DBLog(logs_data=message)
            my_crud = DbLoggerCrud(self.connection)
            my_crud.initiate()
            my_crud.insert(instances=new_log)
        except Exception as e:
            print(e)
