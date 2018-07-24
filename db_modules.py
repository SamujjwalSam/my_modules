#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__synopsis__    : Tools for database operations
__description__ :
__project__     : my_modules
__author__      : 'Samujjwal Ghosh'
__version__     :
__date__        : June 2018
__copyright__   : "Copyright (c) 2018"
__license__     : "Python"; (Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html)

__classes__     :

__variables__   :

__methods__     :

TODO            : 1.
"""

import os
import sqlite3
import my_modules as mm

date_time_tag = mm.get_date_time_tag()


def connect_sqllite(dataset_name,db_name):
    print("Method: connect_sqllite(dataset_name,db_name)")
    path = os.path.join(mm.get_dataset_path(),dataset_name,db_name)
    print("db path: ", path)
    try:
        db = sqlite3.connect(path)
    except sqlite3.Error as er:
        print ('sqlite3 connection error:', er.message)
        return False
    return db


def read_sqllite(db,table_name,cols="*",fetch_one=False):
    print("Method: read_sqllite(db,table_name,cols=\"*\",fetch_one=False)")
    query = "SELECT "+cols+" FROM " + table_name
    print("SQL: ",query)
    cursor = db.cursor()
    cursor.execute(query)
    if fetch_one:
        row_one = cursor.fetchone() # Retrieve the first row
        print(row_one) # Print the first column retrieved(user's name)
    all_rows = cursor.fetchall()
    # for row in all_rows:
        # print('{0} : {1}, {2}'.format(row[0], row[1], row[2]))
    return all_rows


def get_db_details(db):
    print("Method: get_db_details(db)")
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    print("SQL: ",query)
    cursor = db.cursor()
    cursor.execute(query)
    tables = cursor.fetchall()
    print("Tables:",tables)

    import pandas as pd
    for table_name in tables:
        table_name = table_name[0]
        table = pd.read_sql_query("SELECT * from %s" % table_name, db)
        print("writing file: ",table_name + '.csv')
        table.to_csv(table_name + '.csv', encoding='utf-8')
        # print(table.to_csv(table_name + '.csv', index_label='index'))
        # print(table_name+": ",table)
        mm.write_file(table, table_name + '.csv', mode='w', tag=False)


def main():
    pass


if __name__ == "__main__": main()
