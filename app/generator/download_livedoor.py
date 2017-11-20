import os
import urllib.request
import tarfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, 'data')
LIVEDOOR_DIR = os.path.join(DATA_DIR, 'livedoor')
LIVEDOOR_PATH = os.path.join(DATA_DIR, 'livedoor.tar.gz')

url = 'https://www.rondhuit.com/download/ldcc-20140209.tar.gz'
if not os.path.exists(LIVEDOOR_PATH):
  urllib.request.urlretrieve(url, LIVEDOOR_PATH)
  arc_file = tarfile.open(LIVEDOOR_PATH)
  arc_file.extractall(LIVEDOOR_DIR)
  arc_file.close()


