#!/usr/bin/env python


# call user.lookup api to query a list of user ids.
import pandas as pd
import json

pd.set_option('display.max_columns', None)


with open('data.json') as f:
  data = json.load(f)
# Output: {'name': 'Bob', 'languages': ['English', 'Fench']}
df = pd.DataFrame.from_dict(data, orient='columns')
df.set_index("id", inplace=True)
# print(df.head())

df_csv = pd.read_csv('data.csv')
df_csv.set_index("id", inplace=True)

# print(df_csv.head())
result = pd.merge(df,
                  df_csv,
                  on='id')
# print(result.head())

result.to_csv ('final_data_set.csv', header=True)
