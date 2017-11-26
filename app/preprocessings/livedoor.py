import glob
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIVEDOOR_DIR = os.path.join(BASE_DIR, 'data', 'livedoor')

def load_df():
  category = {
    'dokujo-tsushin': 1,
    'it-life-hack':2,
    'kaden-channel': 3,
    'livedoor-homme': 4,
    'movie-enter': 5,
    'peachy': 6,
    'smax': 7,
    'sports-watch': 8,
    'topic-news':9
  }
  docs  = []
  labels = []

  for c_name, c_id in category.items():
    files = glob.glob(LIVEDOOR_DIR + "/text/{c_name}/{c_name}*.txt".format(c_name=c_name))

    text = ''
    for file in files:
      with open(file, 'r') as f:
        lines = f.read().splitlines()

        url = lines[0]
        datetime = lines[1]
        subject = lines[2]
        body = "\n".join(lines[3:])
        text = subject + "\n" + body

      docs.append(text)
      labels.append(c_id)
  df = pd.DataFrame(data = { 'docs': docs, 'labels': labels })
  np.random.seed(0)
  return df.reindex(np.random.permutation(df.index))
