import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

def eda_heatmap(train_data):
    # Target과 상관계수 확인 - heatmap
    plt.figure(figsize=(16, 20))
    matrix = np.triu(train_data.corr())
    sns.heatmap(train_data.corr(),
                annot=True, fmt='.2g',
                mask=matrix,
                vmin=-1, vmax=1, center=0,
                cmap=sns.diverging_palette(20, 220, n=256));
    plt.show()

def eda_vif(X):
    ## 4. VIF 계산
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    sns.barplot(x='VIF', y='Variable', data=vif, color='skyblue')
    plt.show()