##############################################################################
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

def load_df(data_name):
    f_lock = f"data/{data_name}/{data_name}_all.csv"
    df = pd.read_csv(f_lock)
    return df

#####################################################

data_name_lst = ["bace_reg", "esol", "freesolv", "lipo"]

#####################################################

fig, ax = plt.subplots()   # Create a new figure and axis for the plot

# Iterate over the list of dataframes and plot the kernel density estimates for each
for df_name in data_name_lst:
    df = load_df(df_name)
    df_namex = deal_with_names(df_name)
    sns.kdeplot(data=df['y_true'], ax=ax, lw=5, label=df_namex)

# Add axis labels and legend
ax.set_xlabel('\nLabels\n')
ax.set_ylabel('\nKernel Density Estimation\n')
ax.legend()

plt.title('\nHomogeneity of Regression Tasks\n', fontsize=70)

# Show the plot
plt.show()

##############################################################################
##############################################################################
##############################################################################





















