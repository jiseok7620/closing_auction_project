import pandas as pd
import numpy as np

# 데이터 전처리
def pre_process0(df):
    '''
    1. 기본 전처리
    '''
    # NaN 제거
    df = df[~df['imbalance_size'].isna()]

    # 컬럼 제거
    cols = [c for c in df.columns if c not in ['row_id', 'time_id']]
    df = df[cols]

    # NaN이 있으면 0으로 채우기
    df.fillna(0, inplace=True)
    return df

def pre_process1(df):
    '''
    1. 기본 전처리
    2. 다중공선성이 큰 Feature 제거('reference_price', 'bid_price', 'ask_price')
    '''
    # NaN 제거
    df = df[~df['imbalance_size'].isna()]

    # 컬럼 제거
    cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'reference_price', 'bid_price', 'ask_price']]
    df = df[cols]

    # NaN이 있으면 0으로 채우기
    df.fillna(0, inplace=True)
    return df

def pre_process123(df):
    # NaN 제거
    df = df[~df['imbalance_size'].isna()]

    # one-hot encoding
    # 1. 카테고리 컬럼 : imbalance_buy_sell_flag
    df = df.join(pd.get_dummies(df['imbalance_buy_sell_flag'], prefix='imbalance_buy_sell_flag'))
    # 2. 'far_price', 'near_price'가 300초 전 후 이므로 seconds_in_bucket가 300이하 일때와 초과일 때 분류
    # 'A' 컬럼의 값이 300 이하인지 초과인지에 따라 새로운 컬럼 'B' 생성
    df['far_near'] = np.where(df['seconds_in_bucket'] <= 300, 'before_300', 'after_300')
    df = pd.get_dummies(df, columns=['far_near'])

    # 누적값 처리하기
    # 1) 차분(diff)을 활용
    df['matched_diff'] = df.groupby(['stock_id','date_id'])['matched_size'].diff()
    df['matched_diff'] = df['matched_diff'].fillna(0)
    # 2) 로그 변환과 같은 비선형 변환
    df['matched_diff'] = np.log(df['matched_diff'])

    # 상호작용 피쳐 엔지니어링
    df["imb_s1"] = df.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    df["imb_s2"] = df.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    df["price_spread"] = df.eval("ask_price - bid_price")
    df["imbalance_ratio"] = df.eval("imbalance_size / matched_size")
    df['ask_volume'] = df.eval('ask_size * ask_price')
    df['bid_volume'] = df.eval('bid_size * bid_price')
    df['ask_bid_volumes_diff'] = df['ask_volume'] - df['bid_volume']
    df["bid_size_over_ask_size"] = df["bid_size"].div(df["ask_size"])
    df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])

    # 컬럼 제거
    # 1) 'row_id' : object 컬럼, stockid_dateid_secondsinbucket, 카테고리 나눌 필요 X
    # 2) 'time_id' : test 데이터에는 없는 값들 제거 필요
    # 3) 다중공선성 높은 컬럼들 제거 : 'reference_price', 'bid_price', 'ask_price', 'wap' 중에 bid, ask 제거
    # 4) 300초 이후에 제공되는 정보 제거 : 'far_price', 'near_price'
    # 5) 'matched_size' : 누적값으로 파생 feature 만든 후 제거
    # 6) 전처리를 위해 만든 데이터 제거 : 'far_near'
    cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'bid_price', 'ask_price', 'far_price', 'near_price', 'matched_size', 'far_near', 'imbalance_buy_sell_flag']]
    df = df[cols]

    # NaN이 있으면 0으로 채우기
    df.fillna(0, inplace = True)
    return df