import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
import warnings
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

## 1. 데이터 불러오기
warnings.filterwarnings('ignore')
df_train = pd.read_csv("train.csv")

def median_vol_process(df):
    median_vol = pd.DataFrame(columns=['overall_medvol', "first5min_medvol", "last5min_medvol"], index=range(0, 200, 1))
    median_vol['overall_medvol'] = df[['stock_id', "bid_size", "ask_size"]].groupby("stock_id")[["bid_size", "ask_size"]]. \
            median().sum(axis=1).values.flatten()
    median_vol['last5min_medvol'] = df[['stock_id', "bid_size", "ask_size", "far_price", "near_price"]].dropna(). \
            groupby('stock_id')[["bid_size", "ask_size"]].median().sum(axis=1).values.flatten()
    median_vol['first5min_medvol'] = df.loc[(df['far_price'].isna()) | (df['near_price'].isna()),['stock_id', "bid_size", "ask_size"]]. \
            groupby('stock_id')[["bid_size", "ask_size"]].median().sum(axis=1).values.flatten()
    return median_vol

def pre_process(df):
    # NaN 제거
    df = df[~df['imbalance_size'].isna()]

    ### 1-2. far price > 2 인 값 제거 -> 98개 행
    df = df[df['far_price'] < 2]

    # median_vol 추가
    #df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)

    # 1. one-hot encoding
    # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
    #df = df.join(pd.get_dummies(df['imbalance_buy_sell_flag'], prefix='imbalance_buy_sell_flag'))
    df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
    df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
    df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)

    ### 상호작용 feature
    df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
    df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
    df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
    df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
    df['abs_1_price'] = abs(1 - df['wap'])

    # 3. 컬럼 제거
    # 1) 다중공선성 높은 컬럼들 제거 : 'reference_price', 'bid_price', 'ask_price', 'wap' 중 wap 빼고
    cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
    df = df[cols]

    # NaN이 있으면 0으로 채우기
    df.fillna(0, inplace=True)
    return df

def linear_reg():
    # 데이터 일부만 추출
    train_data = df_train.copy()

    ## 데이터 전처리
    #median_vol = median_vol_process(train_data)  # stock 그룹화 컬럼
    #train_data = pre_process(train_data, median_vol)
    train_data = pre_process(train_data)

    ## 특성과 타깃 분할
    # X = stock_data.drop('target', axis=1)
    # y = stock_data['target']
    train_data['target_abs'] = abs(train_data['target'])
    X = train_data.drop(columns=['target', 'target_abs'])
    y = train_data['target_abs']

    ## 학습 세트와 테스트 세트 분할 (여기서는 단순화를 위해 생략)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    ## 선형회귀 모델 학습 (여기서는 전체 데이터로 학습)
    model = LinearRegression()
    model.fit(X_train, y_train)

    ## 해당 모델로 예측하기
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    ## K-Fold CV 객체 생성 (여기서는 k=5)
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    ## 5-Fold CV 수행
    scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_absolute_error')

    print('simple mae : ', mae)
    print('cross mae : ', scores)

def ridge_reg():
    # 데이터 일부만 추출
    train_data = df_train.copy()

    # 종목코드 별로 다른 모델을 적용하기 위한 딕셔너리
    models = {}
    simple = []
    cross = []

    # 종목코드 별로 데이터 분할 및 모델 학습
    for stock_id in train_data['stock_id'].unique():
        ## 해당 종목의 데이터만 선택
        stock_data = train_data[train_data['stock_id'] == stock_id]

        ## 데이터 전처리
        stock_data = pre_process(stock_data)

        ## 특성과 타깃 분할
        X = stock_data.drop('target', axis=1)
        y = stock_data['target']

        ## 스케일링
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        ## 학습 세트와 테스트 세트 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        ## 선형회귀 모델 학습 (여기서는 전체 데이터로 학습)
        penelty = [0.00001, 0.00005, 0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 1, 10]
        model = RidgeCV(alphas=penelty, cv=5)
        model.fit(X_train, y_train)
        print("Best Alpha:{0:.5f}, R2:{1:.4f}".format(model.alpha_, model.best_score_))
        model_best = Ridge(alpha=model.alpha_).fit(X_train, y_train)

        ## 해당 모델로 예측하기
        predictions = model_best.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        simple.append(mae)

        ## K-Fold CV 객체 생성 (여기서는 k=5)
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        scores = cross_val_score(model_best, X, y, cv=kf, scoring='neg_mean_absolute_error') # 5-Fold CV 수행
        cross.append(scores.mean())

        ## models dictionary 에 저장
        print(stock_id)
        models[stock_id] = model

    print('simple mae mean : ', sum(simple) / len(simple))
    print('cross mae mean : ', -(sum(cross) / len(cross)))
    #print(models)

linear_reg()
#ridge_reg()