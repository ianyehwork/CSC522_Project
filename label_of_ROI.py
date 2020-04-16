"""
This program is used for finding the level of roi by percentile and equal size
@author: gwang25
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

#Create label based on the 25%, 50%,75% of return on investment
def create_label_by_percentile(dataframe, chosen_column_name ,new_column_name, percentile_number):
    data = dataframe.copy()
    percentiles = []
    data[new_column_name] = '0'
    value = 0
    percentile_value = []
    percentile_label = []
    percentile_value.append(np.percentile(data[chosen_column_name], value))
    for i in np.arange(percentile_number):
        value += (100/percentile_number)
        percentile_label.append(str(round(value))+"%")
        percentile_value.append(np.percentile(data[chosen_column_name], value))
    percentile_value[0]-=1
    percentile_value[-1] += 1
    print(percentile_value)
    print(percentile_label)
    data[new_column_name] = pd.cut(data[chosen_column_name],
                                   bins=percentile_value,
                                   labels=percentile_label)
    path = "data/movies_meta_data_after_processing_percentile_" + str(percentile_number) + "_label.csv"
    data.to_csv(path)
    return data

def create_label_by_eqaul_range(dataframe, chosen_column_name ,new_column_name, equal_range_number):
    data = dataframe.copy()
    data[new_column_name] = pd.cut(data[chosen_column_name], equal_range_number, precision=2)
    data[new_column_name] = data[new_column_name].astype(str)
    path = "data/movies_meta_data_after_processing_equal_range_" + str(equal_range_number) + "_label.csv"
    data.to_csv(path)
    return data

movies_processed = pd.read_csv('data/movies_meta_data_after_processing.csv')

test_percentile_4=create_label_by_percentile(movies_processed, 'return_on_investment', 'return_on_investment_label',4)
test_percentile_3=create_label_by_percentile(movies_processed, 'return_on_investment', 'return_on_investment_label',3)
test_percentile_4.return_on_investment_label.value_counts()
test_percentile_3.return_on_investment_label.value_counts()


test4=create_label_by_eqaul_range(movies_processed, 'return_on_investment', 'return_on_investment_label',4)
test3=create_label_by_eqaul_range(movies_processed, 'return_on_investment', 'return_on_investment_label',3)
test4.return_on_investment_label.value_counts()
test3.return_on_investment_label.value_counts()


