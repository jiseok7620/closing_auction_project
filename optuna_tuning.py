import sys
import json
import pickle
import numpy as np
from loguru import logger # for nice colored logging
from timeit import default_timer as timer
from pprint import pprint, pformat
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import optuna

def save_model(model, name: str):
    """Saves the LGBM model to disk in {name}.txt and {name}.pkl format.

    Load the pickled model with
    ````
    with open("{name}.pkl", "rb") as f:
         model = pickle.load(f)
    ```
    """
    model.booster_.save_model(f"{name}.txt") # type lgb.basic.Booster

    # and the model with pickle
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(model, f)

def cross_validate(model, X, y, cv): # 교차 검증으로 MAE 평가

    scores = np.zeros(cv.n_splits) # cv.n_splits = 교차 검증 분할 수(n_splits=5 이면 cv.n_splits = 5)
                                   # cv.n_splits = 5 라면, [0. 0. 0. 0. 0.]

    logger.info(f"교차 검증 실행")
    logger.info("="*30)

    for i, (train_index, test_index) in enumerate(cv.split(X)): # 시계열 데이터에 대한 교차 검증 분할 수행
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=0.1)

        start = timer()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
        # eval_set: 검증 세트의 리스트로, 모델이 평가될 때 사용. 여기서는 (X_val, y_val)로 구성된 검증 세트가 제공.
        # callbacks: 콜백 함수의 리스트, 학습 중에 호출되는 함수.
        # lgb.early_stopping(25, verbose=False) : 25로 설정하면 25번의 반복 동안 성능 향상이 없으면 조기 종료. verbose: 조기 종료 메시지를 출력할지 여부를 결정하는 부울 값.
        end = timer()

        y_pred = model.predict(X_test)
        scores[i] = mean_absolute_error(y_pred, y_test)
        logger.info(f"Fold {i+1}: {scores[i]:.4f} (수행시간 : {end - start:.2f}s)")

    logger.success(f"Average MAE = {scores.mean():.4f} ± {scores.std():.2f}")
    logger.info("="*30)

    return scores

def evaluate_simple(model, X, y, cv): # 일반 검증으로 MAE 평가

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # eval sets for early stopping
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=False, test_size=0.2)

    start = timer()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(25, verbose=False)])
    end = timer()

    y_pred = model.predict(X_test)
    score = mean_absolute_error(y_pred, y_test)
    logger.success(f"MAE = {score:.4f} (수행시간 : {end - start:.2f}s)")

    return score


def run_optimization(objective, n_trials=100, n_jobs=1, best_trial=None):
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna의 로깅 출력 레벨을 WARNING으로 설정
    study = optuna.create_study(direction="minimize")  # 하이퍼파라미터 최적화를 위한 Study 객체를 생성
    # 최적화하려는 목적 함수의 방향을 지정
    # - "minimize"으로 설정하면 목적 함수 값을 최소화하는 것을 목표
    # - "maximize"로 설정하면 목적 함수 값을 최대화하는 것을 목표

    if best_trial is not None:  # 이전에 수행한 best_trial이 있다면!
        logger.info("Enqueuing previous best trial ...")
        study.enqueue_trial(best_trial)  # 이전에 수행한 최적의 trial을 현재 Study에 추가
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    logger.info(f"완료된 시행 횟수: {len(study.trials)}")
    logger.success(f"Best MAE: {study.best_value:.4f}")

    logger.info("Params")
    logger.info("=" * 30)
    logger.success(pformat(study.best_params, indent=4))
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f)
    return study

def get_objective_function(LOGGING_LEVELS, X, y, evaluation="simple", cv=None, logging_level="info", ):
    # 예외 처리를 통해 입력된 로깅 레벨이 유효한지 확인하고, 유효하지 않은 경우에는 ValueError 예외를 발생
    if logging_level.upper() not in LOGGING_LEVELS:
        raise ValueError(f"Expected logging_level to be one of {LOGGING_LEVELS}, but got '{logging_level}' instead.")

    # 핸들러를 구성하여 logger 객체가 해당 로깅 레벨로 메시지를 출력하도록 설정
    handler = {"sink": sys.stdout, "level": logging_level.upper()}
    logger.configure(handlers=[handler])

    if evaluation == "simple":  # evaluation="simple" 이면 일반 검증, 아니면 교차검증
        eval_function = evaluate_simple
    else:
        eval_function = cross_validate

    def optimize_lgbm(trial):  # Optuna 라이브러리에서 제공하는 Trial 객체, import optuna 시 사용 가능
        # num_leaves should be smaller than 2^{max_depth}
        max_depth = trial.suggest_int("max_depth", 5, 12)  # max_depth : 17까지
        # max_depth = trial.suggest_int("max_depth", 10, 17)
        num_leaves = trial.suggest_int("num_leaves", 20, int((2 ** max_depth) * 0.75))

        param_space = {
            "objective": trial.suggest_categorical("objective", ["mae"]),
            "random_state": trial.suggest_categorical("random_state", [42]),
            "n_estimators": trial.suggest_categorical("n_estimators", [750]),  # automatically via early stopping
            # "min_child_samples": trial.suggest_int("min_child_samples", 500, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 1, 100),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "subsample_freq": trial.suggest_categorical("subsample_freq", [1]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.8),
            # "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 5e-1, 3.0, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.07, 0.12, log=False),
            # "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.12, log=False),
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_split_gain": trial.suggest_float("min_split_gain", 0, 3),
            "verbosity": -1,
        }
        model = LGBMRegressor(**param_space)
        scores = eval_function(model, X, y, cv=cv)
        return scores.mean()

    return optimize_lgbm