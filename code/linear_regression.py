import utils.utils as utils
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data_dir = utils.data_dir
utils_dir = utils.utils_dir

# First load the lookup tables from the CSV files.
train_product_table = pd.read_csv(utils_dir + "train_offsets.csv")
"""
X = train_product_table["product_id"]
Y = train_product_table["category_id"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0)

linear_regression_model = LinearRegression().fit(x_train, y_train)

y_pred = linear_regression_model.predict(x_test)
print(linear_regression_model.score(x_test, y_test))
print(mean_squared_error(y_test, y_pred))

print(X[-1], Y[-1])

print(train_product_table.corr())
"""


# Assume train_df is dataframe of _id and category_id
freq_df = pd.crosstab(index=train_product_table['category_id'], columns='count').sort_values('count', axis=0, ascending=False)
top10s = dict([(i, train_product_table[train_product_table['category_id'] == i].index) for i in freq_df.index[1500:1520]])
df = pd.DataFrame.from_dict(top10s, orient='index').T
df.plot.hist(stacked=True, bins=100, figsize=(16, 6))
plt.show()
