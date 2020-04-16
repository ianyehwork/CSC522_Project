"""
This program is used for visualiztion which help interpret the statistic of data.
@author: gwang25
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# histogram help us to understand the distribution of attributes
# bins: when bins is an integer, it defines the number of equal_width bins in the range
# densigty: bool, default is false, True: normalization(form a probability density)
def histogram(column_dataframe, bins=10, density=0):
    plt.hist(column_dataframe, bins=bins, density=density, facecolor="blue", edgecolor="black",
             alpha=0.7)
    column_name = column_dataframe.name
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.title("histogram")
    path = 'graph/histogram_'+ column_dataframe.name + '.png'
    plt.savefig(path)
    plt.show()

def boxplot(column_dataframe):
    plt.boxplot(column_dataframe)
    plt.title('Rectangular box plot')
    path = 'graph/boxplot_' + column_dataframe.name + '.png'
    label = column_dataframe.name
    plt.xlabel(label)
    plt.savefig(path)
    plt.show()

# pearson correlation help us understand the correlation between each attributes
def pearson_correlation(dataframe):
    pearsoncorr = dataframe.corr(method='pearson')
    plt.subplots(figsize=(20, 20))
    sns.heatmap(pearsoncorr, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('graph/pearson_correlation.png')
    plt.show()


movies_processed = pd.read_csv('data/movies_meta_data_after_processing.csv')

histogram(movies_processed.return_on_investment, bins=20, density=0)

boxplot(movies_processed.return_on_investment)

pearson_correlation(movies_processed)