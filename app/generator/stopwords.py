import sys, os
# ../preprocessings をimportするおまじない
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessings.stopwords import download

download()

