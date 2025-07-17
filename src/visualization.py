import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import umap
import seaborn as sns

colors_list = [
    'red',
    'olive',
    'steelblue',
    'pink',
    'limegreen',
    'darkgreen',
    'lightcoral',
    'brown',
    'salmon',
    'crimson',
    'peru',
    'peru',
    'peru',
    'tan',
    'salmon',
    'limegreen',
    'skyblue',
    'skyblue',
    'skyblue',
    'mediumorchid',
    'palegreen',
    'peru',
    'seagreen',
    'green',
    'gold',
    'yellow',
    'yellowgreen',
    'yellowgreen',  
    'mediumpurple',
    'indianred',
    'gray',
    'blueviolet',
    'blueviolet',
    'orange']

# plotting function
def plot_curves(dict, dire, name = 'loss_plot', title='Loss Function'):
    
    plt.figure(figsize=(12,8))
    plt.title(title)
    for arg in dict:
        plt.plot(dict[arg], label=arg)
    plt.xlabel("t")
    plt.ylabel(title)
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(dire, name))
    #plt.show()


def tsne_2d(data, **kwargs):
    """
    Transform data to 2d tSNE representation
    :param data: expression data. Shape=(dim1, dim2)
    :param kwargs: tSNE kwargs
    :return:
    """
    print('... performing tSNE')
    tsne = TSNE(n_components=2, **kwargs)
    return tsne.fit_transform(data)

def umap_2d(data, n_neighbors=50, n_components=2, min_dist=0.7, seed=42, **kwargs):
     # Perform UMAP
    umap_model = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=n_components,
                          min_dist=min_dist,
                          random_state=seed, **kwargs)

    # Fit on expression data
    umap_model.fit(data)
    # Get 2D embeddings
    umap_2d = umap_model.transform(data)

    return umap_2d

def plot_tsne_2d(data, labels, epoch, dire, **kwargs):
    """
    Plots tSNE for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    dim1, dim2 = data.shape

    # Prepare label dict and color map
    label_set = set(labels)
    label_dict = {k: v for k, v in enumerate(label_set)}
    print(label_dict)
    # Perform tSNE
    if dim2 == 2:
        # print('plot_tsne_2d: Not performing tSNE. Shape of second dimension is 2')
        data_2d = data
    elif dim2 > 2:
        data_2d = umap_2d(data, **kwargs)
        #data_2d = tsne_2d(data, **kwargs)
    else:
        raise ValueError('Shape of second dimension is <2: {}'.format(dim2))

    # Plot scatterplot
    plt.figure()
    for k, v in label_dict.items():
        
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v)
    plt.legend()
    plt.savefig(os.path.join(dire, 'test_'+ str(epoch)+'.png'))
    return plt.gca()

def scatter_2d(data_2d, labels, colors=None, **kwargs):
    """
    Scatterplot for the provided data, coloring the labels
    :param data: expression data. Shape=(dim1, dim2)
    :param labels: color labels. Shape=(dim1,)
    :param kwargs: tSNE kwargs
    :return: matplotlib axes
    """
    # Prepare label dict and color map
    label_set = list(set(labels))[::-1]
    label_dict = {k: v for k, v in enumerate(label_set)}

    # Plot scatterplot
    i = 0
    for k, v in label_dict.items():
        c = None
        if colors is not None:
            c = colors[i]
        plt.scatter(data_2d[labels == v, 0], data_2d[labels == v, 1],
                    label=v, color=c, **kwargs)
        i += 1
    lgnd = plt.legend(markerscale=3)
    return plt.gca()


def plot_umaps(data_real, data_gen, dire, epoch, tissue_labels = None,  
                n_neighbors=50, n_components=2, min_dist=0.7, seed=42, **kwargs):

    
    full_data = np.concatenate((data_real, data_gen))
    labels = np.array(["real"] * len(data_real) + ["fake"] * len(data_gen))
    #tissue_labels_mapped = [idx_tissue_dict[x] for x in tissue_labels]

    
   # tissue_labels_mapped = {disease: idx for idx, disease in enumerate(tissue_labels)}

    tissue_labels = np.concatenate((tissue_labels, tissue_labels))
    #tissue_labels= tissue_labels_mapped
 
    emb_2d = umap_2d(full_data, n_neighbors=n_neighbors,
                          n_components=n_components,
                          min_dist=min_dist, **kwargs)


    print(emb_2d.shape)
    print(tissue_labels.shape)
    
    
    # plot 1
    #colors =  plt.get_cmap('tab20').colors
    unique_labels = sorted(set(tissue_labels))  # etichette uniche ordinate


    palette = sns.color_palette("hsv", len(unique_labels))


    label_to_color = {label: palette[i] for i, label in enumerate(unique_labels)}


    colors_total = [label_to_color[label] for label in tissue_labels]
    plt.figure(figsize=(18, 6))
    ax = plt.gca()

    plt.subplot(1, 2, 1)
    ax = scatter_2d(emb_2d, tissue_labels, colors=colors_total, 
                    s=10, marker='.')

    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=3, markerscale=5)
    plt.axis('off')

   

    plt.subplot(1, 2, 2)
    colors = ['orange', 'blue']
    emb_2d = emb_2d
    ax = scatter_2d(emb_2d, labels, colors=colors, s=10, marker='.')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=3, markerscale=5)
    plt.axis('off')
    
    plt.subplots_adjust(wspace=0)
    plt.savefig(os.path.join(dire, 'UMAP_combined_' + str(epoch) +'.png'), bbox_extra_artists=(lgd,),  bbox_inches='tight')

    # plot 2
    colors = list(colors_total)
    plt.figure(figsize=(18, 9))
    ax = plt.gca()

    plt.subplot(1, 2, 1)
    ax = scatter_2d(emb_2d[:len(data_real)], tissue_labels[:len(data_real)],colors=colors,
                    s=10, marker='.')

    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=3, markerscale=5)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    ax = scatter_2d(emb_2d[len(data_real):], tissue_labels[len(data_real):],colors=colors, 
                    s=10, marker='.')

    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
            fancybox=True, shadow=True, ncol=3, markerscale=5)
    plt.axis('off')
    plt.subplots_adjust(wspace=0)
    plt.savefig(os.path.join(dire, 'UMAP_separate_' + str(epoch) +'.png'), bbox_extra_artists=(lgd,),  bbox_inches='tight')
    plt.close()


def subplots_umaps(
        nb_points=None,
        umap_proj=None,
        labels1=None,
        labels2=None,
        labels3=None,
        save_to: str = None,
        marker: str = '.',
        epoch: int = None):

    """
    Subplots of same UMAP projections with different labels colored.
    """
    # plot it
    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 2, 2, 2])

    ax0 = plt.subplot(gs[0])
    ax0.scatter(nb_points[:, 0], nb_points[:, 1],
                alpha=0.3, color='blue', marker=marker)
    ax0.plot(np.arange(np.max(nb_points[:, 0])), np.arange(
        np.max(nb_points[:, 0])), color='k', ls='dashed')
    ax0.set_xlabel('Nb real points', size=12)
    ax0.set_ylabel('Nb fake points', size=12)
    if epoch is not None:
        ax0.set_title(f'[Epoch {epoch}] \nPoints repartition')
    elif epoch is None:
        ax0.set_title(f'Points repartition')

    ax1 = plt.subplot(gs[1])
    ax1 = scatter_2d(
        umap_proj,
        labels1,
        s=10,
        marker=marker,
        colors=dict_color_real_samples)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=3, markerscale=2)
    ax1.set_title(f'Real/fake', size=16)
    ax1.set_yticks([])
    ax1.set_xticks([])

    ax2 = plt.subplot(gs[2])
    ax2 = scatter_2d(
        umap_proj,
        labels2,
        s=10,
        marker=marker,
        colors=dict_color_tissue_types)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
               fancybox=True, shadow=True, ncol=3, markerscale=2)
    ax2.set_title(f'Tissues types', size=16)
    ax2.set_yticks([])
    ax2.set_xticks([])
    plt.tight_layout()

    if save_to is not None:
        fig.savefig(save_to)
    elif save_to is None:
        plt.show()



def plot_graph(model, dire, epoch):
    adj_matrix = model.adj.cpu().detach().numpy()
    np.save(os.path.join(dire, 'learned_graph' + str(epoch) +'.npy'), adj_matrix)
    plt.figure(figsize=(4, 4))
    im = plt.imshow(adj_matrix, cmap="inferno", interpolation="nearest", vmin=0, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Learned graph")
    plt.axis("off")
    plt.savefig(os.path.join(dire, 'learned_graph' + str(epoch) +'.png'),  bbox_inches='tight')

'''
def plot_tsne(x_real, x_generated):


    concat_df = pd.concat([x_real, x_generated], axis=0)

    tsne = TSNE(n_components=2, verbose=0, random_state=0)
    z = tsne.fit_transform(x_data)
    df = pd.DataFrame()
    df["y"] = y_data
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    plt.figure(figsize=(4, 4))
    sns.scatterplot(
    x="comp-1",
    y="comp-2",
    hue=df.y.tolist(),
    palette=sns.color_palette("hls", 2),
    alpha=0.5,
    data=df,
    ).set(title=f"T-SNE Projection")
    plt.xlabel("comp-1")
    plt.ylabel("comp-2")
    plt.legend()

    plt.show()


     print('----------Testing----------')
            for i, data in enumerate(test_data):
                
                x = data[0].to(self.device)
                z = data[1].to(self.device)
                generated_samples_x = self.generate_samples(z) 



'''

def plot_distribution(data, label, color='royalblue', linestyle='-', ax=None, plot_legend=True,
                      xlabel=None, ylabel=None):
    """
    Plot a distribution
    :param data: data for which the distribution of its flattened values will be plotted
    :param label: label for this distribution
    :param color: line color
    :param linestyle: type of line
    :param ax: matplotlib axes
    :param plot_legend: whether to plot a legend
    :param xlabel: label of the x axis (or None)
    :param ylabel: label of the y axis (or None)
    :return matplotlib axes
    """
    x = np.ravel(data)
    ax = sns.displot(x,
                      hist=False,
                      kde_kws={'linestyle': linestyle, 'color': color, 'linewidth': 2, 'bw': .15},
                      label=label,
                      ax=ax)
    if plot_legend:
        plt.legend()
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    return ax

def plot_individual_distrs(x, y, symbols, nrows=4, xlabel='X', ylabel='Y'):
    """
    Plots individual distributions for each gene
    """
    nb_symbols = len(symbols)
    ncols = 1 + (nb_symbols - 1) // nrows

    # plt.figure(figsize=(18, 12))
    plt.subplots_adjust(left=0, bottom=0, right=None, top=1.3, wspace=None, hspace=None)
    for r in range(nrows):
        for c in range(ncols):
            idx = (nrows - 1) * r + c
            plt.subplot(nrows, ncols, idx + 1)

            plt.title(symbols[idx])
            plot_distribution(x[:, idx], xlabel='', ylabel='', label=xlabel, color='black')
            plot_distribution(y[:, idx], xlabel='', ylabel='', label=ylabel, color='royalblue')

            if idx + 1 == nb_symbols:
                break

def plot_distance_matrix(dist_m, v_min, v_max, symbols, title='Distance matrix'):
    ax = plt.gca()
    im = ax.imshow(dist_m, vmin=v_min, vmax=v_max)

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(symbols)))
    ax.set_yticks(np.arange(len(symbols)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(symbols)
    ax.set_yticklabels(symbols)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = ax.text(j, i, '{:.2f}'.format(dist_m[i, j]),
                           ha="center", va="center", color="w")
    ax.set_title(title)

def pearson_correlation(x: np.array, y: np.array):
    """
    Computes similarity measure between each pair of genes in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): Gene matrix 1. Shape=(nb_samples, nb_genes_1)
        y (np.array): Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    Returns:
        Matrix with shape (nb_genes_1, nb_genes_2) containing the similarity coefficients
    """

    def standardize(a):
        a_off = np.mean(a, axis=0)
        a_std = np.std(a, axis=0)
        #print(np.sum(a_std==0))
        S = (a - a_off) / a_std
        S[np.isnan(S)] = (a - a_off)[np.isnan(S)]
        return S

def plot_distance_matrices(x, y, symbols, corr_fn=pearson_correlation):
    """
    Plots distance matrices of both datasets x and y.
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :symbols: array of gene symbols. Shape=(nb_genes,)
    :param corr_fn: 2-d correlation function
    """

    dist_x = np.abs(1 - np.abs(corr_fn(x, x)))
    dist_y = np.abs(1 - np.abs(corr_fn(y, y)))
    v_min = min(np.min(dist_x), np.min(dist_y))
    v_max = min(np.max(dist_x), np.max(dist_y))

    # fig = plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plot_distance_matrix(dist_x, v_min, v_max, symbols, title='Distance matrix, real')
    plt.subplot(2, 1, 2)
    plot_distance_matrix(dist_y, v_min, v_max, symbols, title='Distance matrix, synthetic')
    # fig.tight_layout()
    return plt.gca()

