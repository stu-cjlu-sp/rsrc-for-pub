import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sns
import pandas as pd


def plot_confusion_matrix(cm, database,SNR , labels=[]):
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='.2f', xticklabels=labels, yticklabels=labels, cbar=False,
                square=True,
                annot_kws={"fontsize": 20})
    plt.title(database+' SNR='+str(SNR)+'dB')
    plt.xticks(fontsize=20, rotation=45)
    plt.yticks(fontsize=20, rotation=0)
    plt.tight_layout()
    plt.savefig(f'/home/sp432syl/sp432syl/MILAFormer/plotCM/MILAFormer/{database}_{SNR}.pdf', bbox_inches='tight', dpi=450)
    plt.close()
