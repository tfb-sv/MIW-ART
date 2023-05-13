#############################################################################
##############################################################################
##############################################################################

cd C:\Users\nural\OneDrive\Masaüstü\ART-Mol

##############################################################################

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.plot_epoch_curves import deal_with_names 
sns.set_theme(rc={'figure.figsize':(27, 27)}, font_scale=3.5, style="whitegrid")

##############################################################################
##############################################################################
##############################################################################

def count_values(data_names):
    counts = []
    data_namesx = []
    for data_name in data_names:
        file_path = f"data/{data_name}/{data_name}_all.csv"
        df = pd.read_csv(file_path)
        counts.append([sum(df['y_true'] == 0), sum(df['y_true'] == 1)])
        data_namex = deal_with_names(data_name)
        data_namesx.append(data_namex)
    counts_df = pd.DataFrame(counts, columns=["# of 0s", "# of 1s"], index=data_namesx)
    return counts_df

#####################################################

data_names_lst = ["bace", "bbbp2k", "bbbp8k", "clintox", "sider", "tox21"]

counts_df = count_values(data_names_lst)

##############################################################################

sns.heatmap(counts_df, cmap='Blues', annot=True, fmt='g')

# Add a title
plt.title('\nHomogeneity of Classification Tasks\n', fontsize=70)

# Show the plot
plt.show()

##############################################################################
##############################################################################
##############################################################################






























