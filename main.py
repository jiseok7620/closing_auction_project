import pandas as pd
import numpy as np
import json
import warnings
import optuna_tuning
import preprocessing
import eda
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
from loguru import logger
from IPython.display import clear_output
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

## 1. 데이터 불러오기
warnings.filterwarnings('ignore')
df_train = pd.read_csv("train.csv")

## 2. 데이터 전처리
train_data = preprocessing.pre_process1(df_train)

## 3. Data Split
X = train_data.drop(columns='target')
y = train_data['target']

## 4. EDA : target heatmap, feature VIF
eda.eda_heatmap(train_data)
eda.eda_vif(X)

## 5. Best parameter 가져오기 (JSON)
file_path = 'best_params.json'
with open(file_path, 'r') as file:
    data = json.load(file)

## 6. 하이퍼파라미터 튜닝
LOGGING_LEVELS = ["DEBUG", "WARNING", "INFO", "SUCCESS", "WARNING"]
run_lgbm_optimization = True # 최적화를 실행할지 아니면 이미 계산된 최적화를 사용할지 여부.
n_trials = 25 #  샘플링하려는 시행 횟수.
n_jobs = 1 # 사용할 core 수
logging_level = "info" # 평가 함수 내부의 로깅 수준을 구성('info' 또는 'success' 사용).
evaluation = "simple" # 'simple' : 일반검증, or not : 교차검증
cv = TimeSeriesSplit(n_splits=3) # cv: TimeSeriesSplit 분할 객체, n_splits=3(Fold의 개수)
reuse_best_trial = True # reuse_best_trial: 이 최적화를 위해 이전에 조정된 최상의 시도를 재사용할지 여부
lgbm_best_params = data # data or None

if run_lgbm_optimization:
    clear_output(wait=True) # 이전의 출력된 내용을 지움
    objective = optuna_tuning.get_objective_function(LOGGING_LEVELS, X, y, evaluation=evaluation, cv=cv, logging_level=logging_level) # return optimize_lgbm()
    best_trial = lgbm_best_params if reuse_best_trial else None # reuse_best_trial이 True인 경우에만 할당이 이루어지고, 그렇지 않은 경우에는 None으로 설정.
    study = optuna_tuning.run_optimization(objective, n_trials=n_trials, n_jobs=n_jobs, best_trial=best_trial) # objective = optimize_lgbm()
    lgbm_best_params = study.best_params

    fig = plot_optimization_history(study)
    fig.show() # study 객체의 최적화 과정을 시각화하여 히스토리를 그래프로 보여주는 코드
    if n_trials > 1:
        plot_param_importances(study).show()
        plot_parallel_coordinate(study, params=["max_depth", "num_leaves", "learning_rate", "min_split_gain", "min_child_samples"]).show()

## 7. Model prediction
rng = np.random.default_rng()
seeds = rng.integers(low=0, high=1000, size=3)
for i, seed in enumerate(seeds):
    logger.info(f"Fitting model {i} with seed={seed}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = LGBMRegressor(**lgbm_best_params)
    model.set_params(random_state=seed)

    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(25, verbose=True)
    ]
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=callbacks)
    mean_absolute_error(y_test, model.predict(X_test, num_iteration=model.best_iteration_))
    # save_model(model, name=f"model-{i}") # 모델 저장 -> 아직 안함

## 8. Feature Importances
idx = np.argsort(model.feature_importances_)[::-1]
sns.barplot(x=model.feature_importances_[idx], y=np.array(model.feature_name_)[idx], orient="horizontal")
plt.show()