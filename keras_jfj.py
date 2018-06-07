#
# keras example from: https://github.com/jfjensen/numerai_article/blob/master/NumeraiArticle.ipynb


import numpy as np
import pandas as pd


training_data = pd.read_csv('~/numerai_datasets/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('~/numerai_datasets/numerai_tournament_data.csv', header=0)

xx