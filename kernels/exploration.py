from libraries import utilities, visualization
import pandas as pd
import numpy as np
import seaborn as sns
import collections
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA, PCA
from numerai import visualization as numerai_visualization


df_train, df_test = utilities.get_data(n_first_train=-1)
columns = utilities.get_columns(df_train, with_target=False, n_first=-1)


###################### Number of targets
print(dict(collections.Counter(df_train.target)))


###################### Describes dataset
df_description = df_train.describe().T


###################### Classification datasets
visualization.classification_plots(df_train, columns[:10])


###################### PCA test vs PCA train
numerai_visualization.plot_PCA(df_train.drop('target',axis=1), df_test.drop('t_id',axis=1), 'is_test', 'fig', show=True)


###################### Classification datasets
visualization.correlation_matrix(df_train)


###################### Scatter matrix
pd.tools.plotting.scatter_matrix(df_train[columns], alpha=0.2, figsize=(3, 3), diagonal='kde')
pd.tools.plotting.andrews_curves(df_train[columns], 'target')
pd.tools.plotting.lag_plot(df_train[columns])
pd.tools.plotting.radviz(df_train[columns], 'target')


###################### FacetGrid
feature1 = 'feature1'
feature2 = 'feature2'
columns = utilities.get_columns(df_train, with_target=True, n_first=-1)

g = sns.FacetGrid(df_train[columns], row="target")
g.map(plt.hist, feature1, bins=(20))

g = sns.FacetGrid(df_train[columns], row="target")
g.map(sns.kdeplot, feature1)

g = sns.FacetGrid(df_train[columns], row="target")
g.map(plt.scatter, feature1, feature2)

g = sns.FacetGrid(df_train[columns], row="target", margin_titles=True)
g.map(sns.regplot, feature1, feature2, order=2)

g = sns.FacetGrid(df_train[columns], row="target")
g.map(plt.scatter, feature1, feature2)
g.map(sns.kdeplot, feature1, feature2)

sns.lmplot(x=feature1, y='target', data=df_train, logistic=True, y_jitter=.03)


###################### QQPlot
feature = 'feature1'
visualization.boxcox_transformation(df_train, feature, lambda_=None)


###################### PCA
visualization.pca(df_train)


###################### ICA
ica = FastICA(n_components=50)
S_ = ica.fit_transform(df_train.drop('target', axis=1))
A_ = ica.mixing_

pca = PCA(n_components=3)
H = pca.fit_transform(df_train.drop('target', axis=1))

plt.figure()

models = [S_[:,:1], H[:,:1]]
names = [
         'ICA recovered signals',
         'PCA recovered signals',
        ]
colors = ['red', 'blue']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
