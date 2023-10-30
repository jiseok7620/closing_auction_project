import pandas as pd
import numpy as np
import json
import warnings
import optuna_tuning
import preprocessing
import random
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

def pre_process(df):
    # NaN 제거
    df = df[~df['imbalance_size'].isna()]

    # 상호작용 피쳐 엔지니어링
    df["imb_s1"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["imb_s2"] = df.eval("(matched_size-imbalance_size)/(matched_size+imbalance_size)")

    # 컬럼 제거
    # 1) 'row_id' : object 컬럼, stockid_dateid_secondsinbucket, 카테고리 나눌 필요 X
    # 2) 'time_id' : test 데이터에는 없는 값들 제거 필요
    # 3) 'date_id' : test 데이터에는 487부터 시작이라 무의미
    cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id']]
    df = df[cols]

    # NaN이 있으면 0으로 채우기
    df.fillna(0, inplace = True)
    return df

def main():
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.max_rows', 100)

    ## 1. 데이터 불러오기
    warnings.filterwarnings('ignore')
    df_train = pd.read_csv("train.csv")

    # 데이터 일부만 추출
    train_data = df_train.copy()

    # 종목코드 별로 다른 모델을 적용하기 위한 딕셔너리
    models = {}

    # 2. 종목코드 별로 데이터 분할 및 모델 학습
    for stock_id in train_data['stock_id'].unique():
        ## 1) Best parameter 가져오기 (JSON)
        file_path = 'best_params.json'
        with open(file_path, 'r') as file:
            data = json.load(file)

        ## 2) 해당 종목의 데이터만 선택
        stock_data = train_data[train_data['stock_id'] == stock_id]

        ## 3) 데이터 전처리
        stock_data = pre_process(stock_data)

        ## 4) Data Split
        X = stock_data.drop(columns='target')
        y = stock_data['target']

        ## 5) 하이퍼파라미터 튜닝
        LOGGING_LEVELS = ["DEBUG", "WARNING", "INFO", "SUCCESS", "WARNING"]
        run_lgbm_optimization = True # 최적화를 실행할지 아니면 이미 계산된 최적화를 사용할지 여부.
        n_trials = 5 #  샘플링하려는 시행 횟수.
        n_jobs = 1 # 사용할 core 수
        logging_level = "info" # 평가 함수 내부의 로깅 수준을 구성('info' 또는 'success' 사용).
        evaluation = "" # 'simple' : 일반검증, or not : 교차검증
        cv = TimeSeriesSplit(n_splits=3) # cv: TimeSeriesSplit 분할 객체, n_splits=3(Fold의 개수)
        reuse_best_trial = True # reuse_best_trial: 이 최적화를 위해 이전에 조정된 최상의 시도를 재사용할지 여부
        lgbm_best_params = data # data or None

        if run_lgbm_optimization:
            clear_output(wait=True) # 이전의 출력된 내용을 지움
            objective = optuna_tuning.get_objective_function(LOGGING_LEVELS, X, y, evaluation=evaluation, cv=cv, logging_level=logging_level) # return optimize_lgbm()
            best_trial = lgbm_best_params if reuse_best_trial else None # reuse_best_trial이 True인 경우에만 할당이 이루어지고, 그렇지 않은 경우에는 None으로 설정.
            study = optuna_tuning.run_optimization(objective, n_trials=n_trials, n_jobs=n_jobs, best_trial=best_trial) # objective = optimize_lgbm()
            lgbm_best_params = study.best_params

        ## 6) Model prediction
        model = LGBMRegressor(**lgbm_best_params)
        model.fit(X, y)
        models[stock_id] = model
    print(models)

if __name__ == '__main__':
    main()