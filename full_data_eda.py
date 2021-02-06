#%% Load All data 
from load_data import DataLoader

capData = DataLoader()
all_data = capData.load('mainDF')


# %% take a look inside the df
import pandas as pd
pd.set_option('display.max_columns', None)
all_data.head()


# %% check column headers
list(all_data)


# %% see data types

pd.set_option('display.max_rows', None)
all_data.dtypes


#%%isolate one-hot-encoded features

all_data[(all_data == 1 | all_data == 0).any(axis=1)]

# %%
correlation_matrix_data = all_data.select_dtypes(["int","float"])
correlation_matrix_data.shape

# Compute the correlation matrix

corr = correlation_matrix_data.corr()
corr = corr.fillna(0)

# %%
'''
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="white")

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
'''

# %% return only highly correlated features

import seaborn as sns
import matplotlib.pyplot as plt



filteredDf = corr[((corr >= .5) | (corr <= -.5)) & (corr !=1.000)]
plt.figure(figsize=(30,10))
sns.heatmap(filteredDf, annot=True, cmap="Reds")
plt.show()

# %% https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas/17778786
def get_feature_correlation(df, top_n=None, corr_method='spearman',
                            remove_duplicates=True, remove_self_correlations=True):
    """
    Compute the feature correlation and sort feature pairs based on their correlation

    :param df: The dataframe with the predictor variables
    :type df: pandas.core.frame.DataFrame
    :param top_n: Top N feature pairs to be reported (if None, all of the pairs will be returned)
    :param corr_method: Correlation compuation method
    :type corr_method: str
    :param remove_duplicates: Indicates whether duplicate features must be removed
    :type remove_duplicates: bool
    :param remove_self_correlations: Indicates whether self correlations will be removed
    :type remove_self_correlations: bool

    :return: pandas.core.frame.DataFrame
    """
    corr_matrix_abs = df.corr(method=corr_method).abs()
    corr_matrix_abs_us = corr_matrix_abs.unstack()
    sorted_correlated_features = corr_matrix_abs_us \
        .sort_values(kind="quicksort", ascending=False) \
        .reset_index()

    # Remove comparisons of the same feature
    if remove_self_correlations:
        sorted_correlated_features = sorted_correlated_features[
            (sorted_correlated_features.level_0 != sorted_correlated_features.level_1)
        ]

    # Remove duplicates
    if remove_duplicates:
        sorted_correlated_features = sorted_correlated_features.iloc[:-2:2]

    # Create meaningful names for the columns
    sorted_correlated_features.columns = ['Feature 1', 'Feature 2', 'Correlation (abs)']

    if top_n:
        return sorted_correlated_features[:top_n]

    return sorted_correlated_features

# %%
top_corr = get_feature_correlation(corr,50000,remove_duplicates=True, remove_self_correlations=True)
top_corr


# %% Identify

high_corr_features = set(list(top_corr['Feature 2']))

# %% Remove high correlation columns
reduced_data = all_data.drop(high_corr_features,1)

#%% One-hot-encode factors


#%% DRop objects (text fields, dates, person_id, enc_id)

#%% Model

from sklearn.linear_model import LogisticRegression

X = ""
y = ""

lr = LogisticRegression(penalty = 'l1', n_jobs= -1).fit(X, y)
pred = lr.predict()


#%% Preformance

from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score

plot_confusion_matrix(clf, X_test, y_test)  
plt.show()

print("Accuracy: ", accuracy())
print("F1_Score: ", f1_score())
print("Recall: ", recall_score())
print("Precision: ", precision_score())
print("ROC_AUC: ", roc_auc_score())