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
        return None
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


def main():
    pass


if __name__ == "__main__": main()
