---
title: 使用pandas 对比csv 文件
mathjax: true
date: 2024-11-08 13:43:34
categories:
tags: pandas
---

通过pandas 的compare function， 可以对比两个csv 文件
用途，例如升级或修改code之后，输出是csv文本文件，对于同样input的数据，预期应该一样
input -> program -> output
input -> program(optimize) -> output_new
expect output == output_new

```python
import pandas as pd

# Sample data
data_control = {
    'ProductEntityId': [1, 2, 3, 4, 5],
    'ColumnA': [10, 20, 30, 40, 50],
    'ColumnB': ['A', 'B', 'C', 'D', 'E']
}

data_treatment = {
    'ProductEntityId': [1, 2, 3, 4],
    'ColumnA': [10, 25, 30, 40],
    'ColumnB': ['A', 'X', 'C', 'D']
}

df_control = pd.DataFrame(data_control)
df_treatment = pd.DataFrame(data_treatment)

# Set ProductEntityId as the index
df_control.set_index('ProductEntityId', inplace=True)
df_treatment.set_index('ProductEntityId', inplace=True)

# Find the intersection of the indices
common_indices = df_control.index.intersection(df_treatment.index)

# Filter both dataframes to only include the common indices
df_control_common = df_control.loc[common_indices]
df_treatment_common = df_treatment.loc[common_indices]

# Ensure both dataframes have the same columns
df_treatment_common = df_treatment_common[df_control_common.columns]

# Compare the dataframes
comparison_df = df_control_common.compare(df_treatment_common, keep_shape=True, keep_equal=True)

# Highlight the differences
first_level_columns = comparison_df.columns.get_level_values(0).unique()
def highlight_differences(row):
    styles = []
    for col in comparison_df.columns.levels[0]:
        self_val = row[(col, 'self')]
        other_val = row[(col, 'other')]
        if (~pd.isna(self_val) or ~pd.isna(other_val)) & (self_val != other_val):
            styles.append('background-color: yellow')
            styles.append('background-color: yellow')
        else:
            styles.append('')
            styles.append('')
    return styles
highlighted_df = comparison_df.style.apply(highlight_differences, axis=1)

# Display the highlighted dataframe
# highlighted_df
# write hightlighted dataframe to a file
highlighted_df.to_excel("./highlighted_output.xlsx", engine='openpyxl', index=True)

```