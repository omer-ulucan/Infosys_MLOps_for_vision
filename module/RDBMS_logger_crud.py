'''
Copyright 2025-2026 Infosys Ltd.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
""" CRUD Class to handle RDBMS DB based logging """
import sqlalchemy
from sqlalchemy import orm
from module.RDBMS_logger_schema import base


class DbLoggerCrud:
    def __init__(self, connection_string):

        self.connection_string = connection_string
        self.engine = None
        self.session = None

    def initiate(self):
        self.create_engine()
        self.create_session()
        self.create_tables()

    def create_engine(self):
        self.engine = sqlalchemy.create_engine(self.connection_string)

    def create_session(self):
        self.session = sqlalchemy.orm.Session(bind=self.engine)

    def create_tables(self):
        base.metadata.create_all(self.engine)

    def insert(self, instances):
        try:
            self.session.add(instances)
            self.session.commit()
            self.session.flush()
        except Exception as e:
            print(e)
            raise

    def close_session(self):
        try:
            self.session.close()
        except Exception as e:
            print(e)
        else:
            self.session = None

    def close_all_connections(self):
        try:
            self.engine.dispose()
        except Exception as e:
            print(e)
        else:
            self.engine = None
