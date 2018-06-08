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

def create_model(neurons=200, dropout=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(50,), kernel_initializer='glorot_uniform', use_bias=False))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=8, batch_size=128, verbose=0)

neurons = [8, 16]
dropout = [0.01, 0.4]
param_grid = dict(neurons=neurons, dropout=dropout)

gkf = GroupKFold(n_splits=5)
kfold_split = gkf.split(X, Y, groups=complete_training_data.era)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold_split, scoring='neg_log_loss',n_jobs=1, verbose=3)
grid_result = grid.fit(X.values, Y.values)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
