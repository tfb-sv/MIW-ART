#############################################################################
##############################################################################
##############################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

##############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")

##############################################################################
##############################################################################
##############################################################################

def count_values(data_names):
    counts = []
    for data_name in data_names:
        file_path = f"data/{data_name}/{data_name}_all.csv"
        df = pd.read_csv(file_path)
        counts.append([sum(df['y_true'] == 0), sum(df['y_true'] == 1)])
    counts_df = pd.DataFrame(counts, columns=['0s', '1s'], index=data_names)
    return counts_df

#####################################################

data_names_lst = ["bace", "bbbp2k", "clintox", "tox21", "bbbp8k", "sider"]

counts_df = count_values(data_names_lst)

##############################################################################

sns.heatmap(counts_df, cmap='Blues', annot=True, fmt='g')

# Add a title
plt.title('Homogeneity of CLF Tasks')

# Show the plot
plt.show()

##############################################################################
##############################################################################
##############################################################################






























