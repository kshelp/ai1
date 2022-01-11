# 교차 검증과 그리드 서치
# 검증 세트
import pandas as pd

wine = pd.read_csv('https://bit.ly/wine-date')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

print(train_input.shape, test_input.shape)
# (5197, 3) (1300, 3)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)
# (4157, 3) (1040, 3)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
# 0.9971133028626413
# 0.864423076923077


# 교차 검증
from sklearn.model_selection import cross_validate

scores = cross_validate(dt, train_input, train_target)
print(scores)
'''
{'fit_time': array([0.00338101, 0.00509143, 0.00423336, 0.00427008, 0.00527644]), 'score_time': array([0.        , 0.        
, 0.        , 0.00060987, 0.00088668]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])} 
'''

import numpy as np

print(np.mean(scores['test_score']))
# 0.855300214703487

from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
# 0.855300214703487

splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
# 0.8574181117533719


# 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))
# 0.9615162593804117

print(gs.best_params_)
# {'min_impurity_decrease': 0.0001}

print(gs.cv_results_['mean_test_score'])
# [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index])
# {'min_impurity_decrease': 0.0001}

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1),
          'min_samples_split': range(2, 100, 10)
          }

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

print(gs.best_params_)
# {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}

print(np.max(gs.cv_results_['mean_test_score']))
# 0.8683865773302731


# 램덤 서치
from scipy.stats import uniform, randint

rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)

params = {'min_impurity_decrease': uniform(0.0001, 0.001),
          'max_depth': randint(20, 50),
          'min_samples_split': randint(2, 25),
          'min_samples_leaf': randint(1, 25),
          }

from sklearn.model_selection import RandomizedSearchCV

gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params,
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)

print(gs.best_params_)
# {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}

print(np.max(gs.cv_results_['mean_test_score']))
# 0.8695428296438884

dt = gs.best_estimator_

print(dt.score(test_input, test_target))
# 0.86
