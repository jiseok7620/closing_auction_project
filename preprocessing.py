import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from itertools import combinations
from sklearn.decomposition import PCA

# 이상치 처리
def err_process(df):
    def q1_iqr(x):
        q1 = x.quantile(0.25)
        iqr = x.quantile(0.75) - x.quantile(0.25)
        return q1 - (1.5 * iqr)

    def q3_iqr(x):
        q3 = x.quantile(0.75)
        iqr = x.quantile(0.75) - x.quantile(0.25)
        return q3 + (1.5 * iqr)

    # exe
    iqr_df = pd.DataFrame(index=range(0, 200, 1))

    # size
    iqr_df['imbalance_size_errmax'] = df.groupby('stock_id')['imbalance_size'].apply(q3_iqr)
    iqr_df['matched_size_errmax'] = df.groupby('stock_id')['matched_size'].apply(q3_iqr)
    iqr_df['bid_size_errmax'] = df.groupby('stock_id')['bid_size'].apply(q3_iqr)
    iqr_df['ask_size_errmax'] = df.groupby('stock_id')['ask_size'].apply(q3_iqr)

    # price
    iqr_df['reference_price_errmin'] = df.groupby('stock_id')['reference_price'].apply(q1_iqr).astype(np.float32)
    iqr_df['reference_price_errmax'] = df.groupby('stock_id')['reference_price'].apply(q3_iqr).astype(np.float32)
    iqr_df['bid_price_errmin'] = df.groupby('stock_id')['bid_price'].apply(q1_iqr).astype(np.float32)
    iqr_df['bid_price_errmax'] = df.groupby('stock_id')['bid_price'].apply(q3_iqr).astype(np.float32)
    iqr_df['ask_price_errmin'] = df.groupby('stock_id')['ask_price'].apply(q1_iqr).astype(np.float32)
    iqr_df['ask_price_errmax'] = df.groupby('stock_id')['ask_price'].apply(q3_iqr).astype(np.float32)
    iqr_df['wap_errmin'] = df.groupby('stock_id')['wap'].apply(q1_iqr).astype(np.float32)
    iqr_df['wap_errmax'] = df.groupby('stock_id')['wap'].apply(q3_iqr).astype(np.float32)

    # median
    # iqr_df['reference_price_overallmid'] = df.groupby('stock_id')['reference_price'].median().values.flatten().astype(np.float32)
    # iqr_df['bid_price_overallmid'] = df.groupby('stock_id')['bid_price'].median().values.flatten().astype(np.float32)
    # iqr_df['ask_price_overallmid'] = df.groupby('stock_id')['ask_price'].median().values.flatten().astype(np.float32)
    # iqr_df['overall_wapmed'] = df.groupby('stock_id')['wap'].median().values.flatten().astype(np.float32)

    # far, near는 300초 이상, 이하 구분
    # iqr_df['near_price_errmin'] = df[['stock_id', "near_price"]].dropna().groupby('stock_id')['near_price'].apply(q1_iqr).astype(np.float32)
    # iqr_df['near_price_errmax'] = df[['stock_id', "near_price"]].dropna().groupby('stock_id')['near_price'].apply(q3_iqr).astype(np.float32)
    # iqr_df['near_price_overallmid'] = df[['stock_id', "near_price"]].dropna().groupby('stock_id')['near_price'].median().values.flatten().astype(np.float32)
    # iqr_df['far_price_errmin'] = df[['stock_id', "far_price"]].dropna().groupby('stock_id')['far_price'].apply(q1_iqr).astype(np.float32)
    # iqr_df['far_price_errmax'] = df[['stock_id', "far_price"]].dropna().groupby('stock_id')['far_price'].apply(q3_iqr).astype(np.float32)
    # iqr_df['far_price_overallmid'] = df[['stock_id', "far_price"]].dropna().groupby('stock_id')['far_price'].median().values.flatten().astype(np.float32)

    return iqr_df

def median_vol_process(df):
    median_vol = pd.DataFrame(columns=['overall_medvol', "first5min_medvol", "last5min_medvol"], index=range(0, 200, 1))
    median_vol['overall_medvol'] = df[['stock_id', "bid_size", "ask_size"]].groupby("stock_id")[["bid_size", "ask_size"]]. \
            median().sum(axis=1).values.flatten()
    median_vol['last5min_medvol'] = df[['stock_id', "bid_size", "ask_size", "far_price", "near_price"]].dropna(). \
            groupby('stock_id')[["bid_size", "ask_size"]].median().sum(axis=1).values.flatten()
    median_vol['first5min_medvol'] = df.loc[(df['far_price'].isna()) | (df['near_price'].isna()),['stock_id', "bid_size", "ask_size"]]. \
            groupby('stock_id')[["bid_size", "ask_size"]].median().sum(axis=1).values.flatten()

    return median_vol

def drop_col(df):
    # 컬럼 제거
    #cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
    cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id']]
    df = df[cols]
    return  df

def pre_process_reg(df, median_vol, iqr_df):
    ### NaN 제거 -> 220개 행
    df = df[~df['imbalance_size'].isna()]

    ### far price > 2 인 값 제거 -> 98개 행
    # => far, near price 전처리
    df.fillna(0, inplace=True)
    df = df[df['far_price'] < 2]

    ### median_vol 추가 -> 200개 주식 별 bid+ask의 median(전체, 300초 이전, 300초 이후)
    df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)

    ### 이상치 처리 전 size 유의미한 상호작용 피쳐
    df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
    df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
    df['imb_s3'] = df.eval('(imbalance_size-ask_size)/(matched_size+ask_size)').astype(np.float32)
    df['imb_s4'] = df.eval('(imbalance_size-bid_size)/(matched_size+bid_size)').astype(np.float32)

    ### 이상치 처리 (size컬럼들에서)
    df = df.merge(iqr_df, how='left', left_on="stock_id", right_index=True)

    # colums = ['imbalance_size_errmax', 'matched_size_errmax', 'bid_size_errmax', 'ask_size_errmax',
    #     'bid_price_errmin', 'bid_price_errmax', 'ask_price_errmin', 'ask_price_errmax', 'reference_price_errmin', 'reference_price_errmax', 'wap']
    colums = ['imbalance_size_errmax', 'matched_size_errmax', 'bid_size_errmax', 'ask_size_errmax',
              'bid_price_errmin', 'bid_price_errmax', 'ask_price_errmin', 'ask_price_errmax', 'reference_price_errmin',
              'reference_price_errmax', 'wap']

    for i in colums:
        result = i.split("_")
        if i == 'wap':
            # IQT MAX
            df['comparison_result'] = df[f'{i}_errmax'] < df[i]
            true_indexs = df.index[df['comparison_result']].tolist()
            df.loc[df.index.isin(true_indexs), i] = df.loc[df.index.isin(true_indexs), f'{i}_errmax']
            df = df.drop(['comparison_result', f'{i}_errmax'], axis=1)

            # IQT MIN
            df['comparison_result'] = df[f'{i}_errmin'] > df[i]
            true_indexs = df.index[df['comparison_result']].tolist()
            df.loc[df.index.isin(true_indexs), i] = df.loc[df.index.isin(true_indexs), f'{i}_errmin']
            df = df.drop(['comparison_result', f'{i}_errmin'], axis=1)

        elif result[1] == 'size':
            df['comparison_result'] = df[i[0:-7]] > df[i]
            true_indexs = df.index[df['comparison_result']].tolist()
            df.loc[df.index.isin(true_indexs), i[0:-7]] = df.loc[df.index.isin(true_indexs), i]
            df = df.drop(['comparison_result', f'{i}'], axis=1)

        elif result[1] == 'price':
            if i[-6:] == 'errmin':
              df['comparison_result'] = df[i[0:-7]] < df[i]
              true_indexs = df.index[df['comparison_result']].tolist()
              df.loc[df.index.isin(true_indexs), i[0:-7]] = df.loc[df.index.isin(true_indexs), i]
              df = df.drop(['comparison_result', f'{i}'], axis=1)
            else:
              df['comparison_result'] = df[i[0:-7]] > df[i]
              true_indexs = df.index[df['comparison_result']].tolist()
              df.loc[df.index.isin(true_indexs), i[0:-7]] = df.loc[df.index.isin(true_indexs), i]
              df = df.drop(['comparison_result', f'{i}'], axis=1)

    ### 이상치 처리 후 size 상호작용 피쳐
    df['imb_s5'] = df.eval('(ask_size-bid_size)')

    ### 이상치 처리 후 price 상호작용 피쳐
    df['reference_price_plus_bid_price'] = df.eval('(reference_price+bid_price)')
    df['reference_price_plus_askd_price'] = df.eval('(reference_price+bid_size)')

    ### price 유의미한 상호작용 피쳐
    # prices = ['reference_price', 'ask_price', 'bid_price', 'wap']
    # for c in combinations(prices, 2):
    #     df[f'{c[0]}_plus_{c[1]}'] = (df[f'{c[0]}'] + df[f'{c[1]}']).astype(np.float32)
    #     df[f'{c[0]}_X_{c[1]}'] = (df[f'{c[0]}'] * df[f'{c[1]}']).astype(np.float32)
    #
    # for c in combinations(prices, 3):
    #     max_ = df[list(c)].max(axis=1)
    #     min_ = df[list(c)].min(axis=1)
    #     df[f'{c[0]}_{c[1]}_{c[2]}_plus'] = (max_+min_).astype(np.float32)
    #     df[f'{c[0]}_{c[1]}_{c[2]}_X'] = (max_*min_).astype(np.float32)

    # gc.collect()
    gc.collect()

    return df

# def pre_process_clf(df, median_vol):#, iqr_df):
#     ### NaN 제거 -> 220개 행
#     df = df[~df['imbalance_size'].isna()]
#
#     ### median_vol 추가 -> 200개 주식 별 bid+ask의 median(전체, 300초 이전, 300초 이후)
#     df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)
#
#     ### feature
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
#     df['bid_minus_ask_sizes'] = df['bid_size'] - df['ask_size']
#
#     ### NaN이 있으면 0으로 채우기
#     # => far, near price 전처리
#     df.fillna(0, inplace=True)
#
#     ### far price > 2 인 값 제거 -> 98개 행
#     df = df[df['far_price'] < 2]
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process_inte(df, median_vol):
#     ### NaN 제거 -> 220개 행
#     df = df[~df['imbalance_size'].isna()]
#
#     ### median_vol 추가 -> 200개 주식 별 bid+ask의 median(전체, 300초 이전, 300초 이후)
#     df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)
#
#     ### clf feature
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)')
#     df['bid_minus_ask_sizes'] = df['bid_size'] - df['ask_size']
#
#     ### reg feature
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)')
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price'])
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price'])
#     df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
#     df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
#     df['abs_1_price'] = abs(1 - df['wap'])
#
#     ### NaN이 있으면 0으로 채우기 -> far, near price 전처리
#     df.fillna(0, inplace=True)
#
#     ### far price > 2 인 값 제거 -> 98개 행
#     df = df[df['far_price'] < 2]
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
#
# # 데이터 전처리
# def pre_process0(df):
#     '''
#     1. 기본 전처리
#     => 6.4164
#
#     # row_id : object 컬럼, 불필요한 컬럼
#     # time_id : test에 없는 컬럼
#     # date_id : test(487~)의 값과 train(~480)값이 아예 동떨어짐
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process1(df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성이 큰 Feature 제거('reference_price', 'bid_price', 'ask_price')
#     => 6.4384
#
#     # VIF가 10이상, corr가 0.95 이상
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if
#             c not in ['row_id', 'time_id', 'date_id', 'reference_price', 'bid_price', 'ask_price']]
#     df = df[cols]
#
#    # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process2(df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     => 6.3983
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # price
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     # imbalance
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process3(df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     4. One-Hot Encoding
#         1) imbalance_buy_sell_flag : -1, 0, 1 값 나누기
#         2) seconds_in_bucket : 300초 이하, 이전 나누기
#         3) imbalance_buy_sell_flag 제거 => 6.3982
#         4) far_price, near_price 제거 => 6.3989
#         5) imbalance_buy_sell_flag 인코딩만 추가, imbalance_buy_sell_flag 제거 => 3.977
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#     # 2) 'far_price', 'near_price'가 300초 전 후 이므로 seconds_in_bucket가 300이하 일때와 초과일 때 분류
#     # 'A' 컬럼의 값이 300 이하인지 초과인지에 따라 새로운 컬럼 'B' 생성
#     # df['before_300'] = df['seconds_in_bucket'].apply(lambda x: 1 if x < 300 else 0)
#     # df['after_300'] = df['seconds_in_bucket'].apply(lambda x: 1 if x >= 300 else 0)
#
#     # price
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     # imbalance
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process4(df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     4. One-Hot Encoding
#         1) imbalance_buy_sell_flag : -1, 0, 1 값 나누기
#         2) seconds_in_bucket : 300초 이하, 이전 나누기 => X
#     5. 누적값 추가하기 : matched_diff
#         1) matched_size 살리기
#         2) matched_size 제거
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#
#     # 누적값 처리하기
#     # 1) 차분(diff)을 활용
#     df['matched_diff'] = df.groupby(['stock_id', 'date_id'])['matched_size'].diff()
#     df['matched_diff'] = df['matched_diff'].fillna(0)
#
#     # price
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     # imbalance
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process5(df, median_vol):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     4. One-Hot Encoding
#         1) imbalance_buy_sell_flag : -1, 0, 1 값 나누기
#         2) seconds_in_bucket : 300초 이하, 이전 나누기 => X
#     5. 누적값 추가하기 : matched_diff
#         1) matched_size 살리기 => O
#         2) matched_size 제거 => X
#     6. median_vol 추가
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # median_vol 추가
#     df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)
#
#     # one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#
#     # 누적값 처리하기
#     # 1) 차분(diff)을 활용
#     df['matched_diff'] = df.groupby(['stock_id', 'date_id'])['matched_size'].diff()
#     df['matched_diff'] = df['matched_diff'].fillna(0)
#
#     # price
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     # imbalance
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process6(df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     4. One-Hot Encoding
#         1) imbalance_buy_sell_flag : -1, 0, 1 값 나누기
#         2) seconds_in_bucket : 300초 이하, 이전 나누기 => X
#     5. 누적값 추가하기 : matched_diff
#         1) matched_size 살리기 => O
#         2) matched_size 제거 => X
#     6. PCA 추가
#     '''
#     # NaN 제거
#     df = df[~df['imbalance_size'].isna()]
#
#     # PCA
#     prices = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', 'wap']
#     pca_prices = PCA(n_components=1)
#     df['pca_prices'] = pca_prices.fit_transform(df[prices].fillna(1))
#
#     # one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#
#     # 누적값 처리하기
#     # 1) 차분(diff)을 활용
#     df['matched_diff'] = df.groupby(['stock_id', 'date_id'])['matched_size'].diff()
#     df['matched_diff'] = df['matched_diff'].fillna(0)
#
#     # price
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     # imbalance
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     # 컬럼 제거
#     cols = [c for c in df.columns if c not in ['row_id', 'time_id', 'date_id', 'imbalance_buy_sell_flag']]
#     df = df[cols]
#
#     # NaN이 있으면 0으로 채우기
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# def pre_process8(df, median_vol, iqr_df):
#     ### 1. NaN 제거 -> 220개 행
#     df = df[~df['imbalance_size'].isna()]
#
#     ### 2. median_vol 추가 -> 200개 주식 별 bid+ask의 median(전체, 300초 이전, 300초 이후)
#     df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)
#
#     ### 3. 이상치 처리 (size컬럼들에서)
#     df = df.merge(iqr_df, how='left', left_on="stock_id", right_index=True)
#
#     colums = ['imbalance_size_errmax', 'matched_size_errmax', 'reference_price_errmin', 'reference_price_errmax']
#
#     for i in colums:
#         if i[-6:] == 'errmin':
#             df['comparison_result'] = df[i[0:-7]] < df[i]
#             true_indexs = df.index[df['comparison_result']].tolist()
#             df.loc[df.index.isin(true_indexs), i[0:-7]] = df.loc[df.index.isin(true_indexs), i]
#             df = df.drop(['comparison_result', f'{i}'], axis=1)
#         else:
#             df['comparison_result'] = df[i[0:-7]] > df[i]
#             true_indexs = df.index[df['comparison_result']].tolist()
#             df.loc[df.index.isin(true_indexs), i[0:-7]] = df.loc[df.index.isin(true_indexs), i]
#             df = df.drop(['comparison_result', f'{i}'], axis=1)
#
#     ### 4. one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#
#     ### 5. 누적값 처리하기
#     # 1) 차분(diff)을 활용
#     df['matched_diff'] = df.groupby(['stock_id', 'date_id'])['matched_size'].diff()
#     df['matched_diff'] = df['matched_diff'].fillna(0)
#
#     ### 6. price 상호작용 feature
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     ### 7. imbalance 상호작용 feature
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     ### WAP와 연관된 feature들 추가
#     df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
#     df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
#     df['ask_size_x_bid_price'] = df.eval('ask_size*bid_price')
#     df['bid_size_x_ask_price'] = df.eval('bid_size*ask_price')
#     df['ask_x_size'] = df.eval('ask_size*ask_price')
#     df['bid_x_size'] = df.eval('bid_size*bid_price')
#     df['ask_minus_bid'] = df['ask_x_size'] - df['bid_x_size']
#     df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
#
#     ### Target값이 양수인지 음수인지 구별을 잘하게 하는 feature
#
#     ### wap과 1과의 거리
#     df['abs_1_price'] = abs(1 - df['wap'])
#
#     ### 8. NaN이 있으면 0으로 채우기
#     # => far, near price 전처리
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
#
# def pre_process7(df, median_vol, iqr_df):
#     '''
#     1. 기본 전처리
#     2. 다중공선성 제거 X
#     3. 상호작용 Feature 중 Importance가 높은 것들만 추가
#     4. One-Hot Encoding
#         1) imbalance_buy_sell_flag : -1, 0, 1 값 나누기
#         2) seconds_in_bucket : 300초 이하, 이전 나누기 => X
#     5. 누적값 추가하기 : matched_diff
#         1) matched_size 살리기 => O
#         2) matched_size 제거 => X
#     6. median_vol 추가
#
#     +++)
#     7. imbalance_size와 matched_size의 ratio는 의미가 있을까?
#         1) matched_size는 음수가 없지만 imbalance_size는 0이 있으므로, imbalance_size/matched_size로 산출
#     '''
#     ### 1. NaN 제거 -> 220개 행
#     df = df[~df['imbalance_size'].isna()]
#
#     ### 2. median_vol 추가 -> 200개 주식 별 bid+ask의 median(전체, 300초 이전, 300초 이후)
#     df = df.merge(median_vol, how='left', left_on="stock_id", right_index=True)
#
#     ### 3. 이상치 처리 (size컬럼들에서)
#     df = df.merge(iqr_df, how='left', left_on="stock_id", right_index=True)
#     df['comparison_result1'] = df['imbalance_size'] > df['imbalance_size_errmax']
#     true_indexs = df.index[df['comparison_result1']].tolist()
#     df.loc[df.index.isin(true_indexs), 'imbalance_size'] = df.loc[df.index.isin(true_indexs), 'imbalance_size_errmax']
#
#     df['comparison_result2'] = df['matched_size'] > df['matched_size_errmax']
#     true_indexs = df.index[df['comparison_result2']].tolist()
#     df.loc[df.index.isin(true_indexs), 'matched_size'] = df.loc[df.index.isin(true_indexs), 'matched_size_errmax']
#
#     ### 4. one-hot encoding
#     # 1) 카테고리 컬럼 : imbalance_buy_sell_flag
#     df['imbalance_buy_sell_flag_m'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == -1 else 0)
#     df['imbalance_buy_sell_flag_0'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 0 else 0)
#     df['imbalance_buy_sell_flag_1'] = df['imbalance_buy_sell_flag'].apply(lambda x: 1 if x == 1 else 0)
#
#     ### 5. 누적값 처리하기
#     # 1) 차분(diff)을 활용
#     df['matched_diff'] = df.groupby(['stock_id', 'date_id'])['matched_size'].diff()
#     df['matched_diff'] = df['matched_diff'].fillna(0)
#
#     ### 6. price 상호작용 feature
#     df['bid_price-reference_price'] = (df['bid_price'] - df['reference_price']).astype(np.float32)
#     df['ask_price-reference_price'] = (df['ask_price'] - df['reference_price']).astype(np.float32)
#
#     ### 7. imbalance 상호작용 feature
#     df['imb_s1'] = df.eval('(bid_size-ask_size)/(bid_size+ask_size)').astype(np.float32)
#     df['imb_s2'] = df.eval('(imbalance_size-matched_size)/(matched_size+imbalance_size)').astype(np.float32)
#
#     ### WAP와 연관된 feature들 추가
#     df['bid_plus_ask_sizes'] = df['bid_size'] + df['ask_size']
#     df['imbalance_ratio'] = df['imbalance_size'] / df['matched_size']
#     df['ask_size_x_bid_price'] = df.eval('ask_size*bid_price')
#     df['bid_size_x_ask_price'] = df.eval('bid_size*ask_price')
#     df['ask_x_size'] = df.eval('ask_size*ask_price')
#     df['bid_x_size'] = df.eval('bid_size*bid_price')
#     df['ask_minus_bid'] = df['ask_x_size'] - df['bid_x_size']
#     df["bid_price_over_ask_price"] = df["bid_price"].div(df["ask_price"])
#
#     ### 8. NaN이 있으면 0으로 채우기
#     # => far, near price 전처리
#     df.fillna(0, inplace=True)
#
#     # gc.collect()
#     gc.collect()
#
#     return df
#
# # stock_id, date_id 그룹화하여 매칭
# def two_group(df):
#     # size 컬럼들을 stock_id, date_id로 그룹화하여 median값 구하기
#     twin_group = df[['stock_id', 'date_id']].groupby(["stock_id", "date_id"]).mean().reset_index()
#     twin_group['imbalance_size_medvol'] = \
#     df[['stock_id', 'date_id', 'imbalance_size']].groupby(["stock_id", "date_id"])['imbalance_size']. \
#         median().values.flatten()
#     twin_group['matched_size_medvol'] = df[['stock_id', 'date_id', 'matched_size']].groupby(["stock_id", "date_id"])[
#         'matched_size']. \
#         median().values.flatten()
#     twin_group['bid_size_medvol'] = df[['stock_id', 'date_id', 'bid_size']].groupby(["stock_id", "date_id"])['bid_size'] \
#         .median().values.flatten()
#     twin_group['ask_size_medvol'] = df[['stock_id', 'date_id', 'ask_size']].groupby(["stock_id", "date_id"])['ask_size'] \
#         .median().values.flatten()
#     df = df.merge(twin_group, on=['stock_id', 'date_id'], how='inner')  # median_vol 추가 => stock_id, date_id 별로 median 값을 추가
#     return df