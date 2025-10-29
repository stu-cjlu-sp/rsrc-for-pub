import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
sns.set(style="white",font_scale=1.0)

def plot_tsne(features, labels,dataset,snr,classes):
    
    sns.set(style="white")
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot()
    color = labels
    if dataset == '22.01A':
        color_map = ["#FF5733", "#05A568", '#9575CD','#29B6F6', '#00CFFF', '#E2526A', '#F37F13','#804A10','#5FFDAC','#B1EA15','#CC3C7F']
    # ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']    
    else:
        color_map = ["#FF5733", "#05A568", '#9575CD','#29B6F6', '#00CFFF', '#E2526A', '#F37F13','#804A10','#5FFDAC','#B1EA15','#CC3C7F']
    # ['8PSK', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'AM-DSB', 'AM-SSB', 'WBFM']    
   
    df = pd.DataFrame()
    df["y"] = labels
    df["comp1"] = X_tsne[:, 0] 
    df["comp2"] = X_tsne[:, 1]

    sns.scatterplot(x= df.comp1.tolist(), y= df.comp2.tolist(),hue=df.y.tolist(),
                    palette=sns.color_palette(color_map,len(color_map)),edgecolor="none",
                    data=df)
    handles, labels = ax.get_legend_handles_labels()    
    ax.legend(handles, classes,fontsize=12,)

    
    plt.title(f't-SNE Visualization SNR = {snr}dB',fontsize=25)
    ax.set_xticks([])  
    ax.set_yticks([])  
    plt.savefig(f'/home/sp432syl/sp432syl/MILAFormer/tsne/MILAFormer/tsne_{dataset}_{snr}.pdf', dpi=450)
    plt.close()

if __name__ == '__main__':
    digits = datasets.load_digits(n_class=11)
    features, labels = digits.data, digits.target
    print(features.shape)
    print(labels.shape)
    plot_tsne(features, labels)
