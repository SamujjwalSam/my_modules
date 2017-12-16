import os

import sys,platform
if platform.system() == 'Windows':
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research')
    sys.path.append('D:\GDrive\Dropbox\IITH\\0 Research\Datasets')
else:
    sys.path.append('/home/cs16resch01001/codes')
    sys.path.append('/home/cs16resch01001/datasets')
print(platform.system(),"os detected.")


def get_date_time_tag(current_file_name=os.path.basename(__file__)):
    from datetime import datetime
    date_time = datetime.now().strftime('%Y%m%d %H%M%S')
    if not current_file_name:
        current_file_name = os.path.basename(__file__)
    return current_file_name+"_"+date_time+"_"


date_time_tag = get_date_time_tag()


def main():
    pass


if __name__ == "__main__": main()
