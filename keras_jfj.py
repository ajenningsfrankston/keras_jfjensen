#
# keras example from: https://github.com/jfjensen/numerai_article/blob/master/NumeraiArticle.ipynb


import numpy as np
import pandas as pd


training_data = pd.read_csv('~/numerai_datasets/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('~/numerai_datasets/numerai_tournament_data.csv', header=0)

validation_data = tournament_data[tournament_data.data_type=='validation']
complete_training_data = training_data

# (note not training on validation data)

features = [f for f in list(complete_training_data) if "feature" in f]
X = complete_training_data[features]
Y = complete_training_data["target"]


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GroupKFold
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier


xxx