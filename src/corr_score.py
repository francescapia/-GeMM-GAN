""" Script for the Correlation Score metric.

Reference:
-----
Ramon Viñas, Helena Andrés-Terré, Pietro Liò, Kevin Bryson,
Adversarial generation of gene expression data,
Bioinformatics, Volume 38, Issue 3, February 2022, Pages 730–737,
https://doi.org/10.1093/bioinformatics/btab035

Original code:
-----
https://github.com/rvinas/adversarial-gene-expression.git

"""

# Imports
import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram

def upper_diag_list(m_: np.array):
    """
    Returns the condensed list of all the values in the upper-diagonal of m_
    ----
    Parameters:
        m_ (np.array): array of float. Shape=(N, N)
    Returns:
        list of values in the upper-diagonal of m_ (from top to bottom and from
             left to right). Shape=(N*(N-1)/2,)
    """
    m = np.triu(m_, k=1)  # upper-diagonal matrix
   
    tril = np.zeros_like(m_) + np.nan
    
    tril = np.tril(tril)

    m += tril
    
    m = np.ravel(m)

    return m[~np.isnan(m)]


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


    assert x.shape[0] == y.shape[0]

    x_ = standardize(x)
    y_ = standardize(y)
   
    return np.dot(x_.T, y_) / x.shape[0]


def gamma_coeff_score(x_test: np.array, x_gen: np.array):
    """
    Compute correlation score for two given expression matrices
    ----
    Parameters:
        x (np.array): matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
        y (np.array): matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    Returns:
        Gamma(D^X, D^Z)
    """
    dists_x = 1 - correlations_list(x_test, x_test)

    dists_y = 1 - correlations_list(x_gen, x_gen)
     

    gamma_dx_dy = pearson_correlation(dists_x, dists_y)

    return gamma_dx_dy


def correlations_list(x: np.array, y: np.array):
    """
    Generates correlation list between all pairs of genes in the bipartite graph x <-> y
    ----
    Parameters:
        x (np.array): Gene matrix 1. Shape=(nb_samples, nb_genes_1)
        y (np.array): Gene matrix 2. Shape=(nb_samples, nb_genes_2)
    Returns:
        corr_fn (np.array): correlation function taking x and y as inputs
    """

    corr = pearson_correlation(x, y)

    return upper_diag_list(corr)

def gamma_coef(x: np.array, y: np.array):
    """
    Compute gamma coefficients for two given expression matrices
    :param x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param y: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z)
    """
 
    dists_x = 1 - correlations_list(x, x)


    dists_y = 1 - correlations_list(y, y)

    gamma_dx_dy = pearson_correlation(dists_x, dists_y)
    return gamma_dx_dy

def hierarchical_clustering(data, corr_fun=pearson_correlation):
    """
    Performs hierarchical clustering to cluster genes according to a gene similarity
    metric.
    Reference: Cluster analysis and display of genome-wide expression patterns
    :param data: numpy array. Shape=(nb_samples, nb_genes)
    :param corr_fun: function that computes the pairwise correlations between each pair
                     of genes in data
    :return scipy linkage matrix
    """
    # Perform hierarchical clustering
    y = 1 - correlations_list(data, data)
    l_matrix = linkage(y, 'complete')  # 'correlation'
    return l_matrix

def gamma_coefficients(expr_x, expr_z):
    """
    Compute gamma coefficients for two given expression matrices
    :param expr_x: matrix of gene expressions. Shape=(nb_samples_1, nb_genes)
    :param expr_z: matrix of gene expressions. Shape=(nb_samples_2, nb_genes)
    :return: Gamma(D^X, D^Z), Gamma(D^X, T^X), Gamma(D^Z, T^Z), Gamma(T^X, T^Z)
             where D^X and D^Z are the distance matrices of expr_x and expr_z (respectively),
             and T^X and T^Z are the dendrogrammatic distance matrices of expr_x and expr_z (respectively).
             Gamma(A, B) is a function that computes the correlation between the elements in the upper-diagonal
             of A and B.
    """
    # Compute Gamma(D^X, D^Z)
    dists_x = 1 - correlations_list(expr_x, expr_x)
    dists_z = 1 - correlations_list(expr_z, expr_z)
    gamma_dx_dz = pearson_correlation(dists_x, dists_z)

    # Compute Gamma(D^X, T^X)
    xl_matrix = hierarchical_clustering(expr_x)
    #gamma_dx_tx, _ = cophenet(xl_matrix, dists_x)

    # Compute Gamma(D^Z, T^Z)
    zl_matrix = hierarchical_clustering(expr_z)
    #gamma_dz_tz, _ = cophenet(zl_matrix, dists_z)

    # Compute Gamma(T^X, T^Z)
    gamma_tx_tz = compare_cophenetic(xl_matrix, zl_matrix)
    # gamma_dx_tx, gamma_dz_tz,
    return gamma_dx_dz, gamma_tx_tz

class Cluster:
    """
    Auxiliary class to store binary clusters
    """

    def __init__(self, c_left=None, c_right=None, index=None):
        assert (index is not None) ^ (c_left is not None and c_right is not None)
        self._c_left = c_left
        self._c_right = c_right
        if index is not None:
            self._indices = [index]
        else:
            self._indices = c_left.indices + c_right.indices

    @property
    def indices(self):
        return self._indices

    @property
    def c_left(self):
        return self._c_left

    @property
    def c_right(self):
        return self._c_right


def dendrogram_distance(l_matrix, condensed=True):
    """
    Computes the distances between each pair of genes according to the scipy linkage
    matrix.
    :param l_matrix: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :param condensed: whether to return the distances as a flat array containing the
           upper-triangular of the distance matrix
    :return: distances
    """
    nb_genes = l_matrix.shape[0] + 1

    # Fill distance matrix m
    clusters = {i: Cluster(index=i) for i in range(nb_genes)}
    m = np.zeros((nb_genes, nb_genes))
    for i, z in enumerate(l_matrix):
        c1, c2, dist, n_elems = z
        clusters[nb_genes + i] = Cluster(c_left=clusters[c1],
                                         c_right=clusters[c2])
        c1_indices = clusters[c1].indices
        c2_indices = clusters[c2].indices

        for c1_idx in c1_indices:
            for c2_idx in c2_indices:
                m[c1_idx, c2_idx] = dist
                m[c2_idx, c1_idx] = dist

    # Return flat array if condensed
    if condensed:
        return upper_diag_list(m)

    return m


def compare_cophenetic(l_matrix1, l_matrix2):
    """
    Computes the cophenic distance between two dendrograms given as scipy linkage matrices
    :param l_matrix1: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :param l_matrix2: Scipy linkage matrix. Shape=(nb_genes-1, 4)
    :return: cophenic distance between two dendrograms
    """
    dists1 = dendrogram_distance(l_matrix1, condensed=True)
    dists2 = dendrogram_distance(l_matrix2, condensed=True)

    return pearson_correlation(dists1, dists2)

def tf_tg_interactions(dir, graph_file):
    """
    Returns a dictionary with TFs as keys and their target genes as values
    :param dir: directory containing the graph_file
    :param graph_file: file containing the TF-TG interactions
    :return: dictionary with TFs as keys and their target genes as values
    """
    tf_tg = {}
    with open(f'{dir}/{graph_file}', 'r') as f:
        for line in f:
            tf, _, tg = line.strip().split('\t')
            if tf not in tf_tg:
                tf_tg[tf] = []
            tf_tg[tf] += [tg]

    return tf_tg
    
def compute_tf_tg_corrs(expr, gene_symbols, tf_tg=None, flat=True):
    """
    Computes the lists of TF-TG and TG-TG correlations
    :param expr: matrix of gene expressions. Shape=(nb_samples, nb_genes)
    :param gene_symbols: list of gene symbols matching the expr matrix. Shape=(nb_genes,)
    :param tf_tg: dict with TF symbol as key and list of TGs' symbols as value
    :param flat: whether to return flat lists
    :return: lists of TF-TG and TG-TG correlations, respectively
    """
    if tf_tg is None:
        tf_tg = tf_tg_interactions()
    gene_symbols = np.array(gene_symbols)

    tf_tg_corr = []
    tg_tg_corr = []
    for tf, tgs in tf_tg.items():
        tg_idxs = np.array([np.where(gene_symbols == tg)[0] for tg in tgs if tg in gene_symbols]).ravel()

        if tf in gene_symbols and len(tg_idxs) > 0:
            # TG-TG correlations
            expr_tgs = expr[:, tg_idxs]
            corr = correlations_list(expr_tgs, expr_tgs)
            tg_tg_corr += [corr.tolist()]

            # TF-TG correlations
            tf_idx = np.argwhere(gene_symbols == tf)[0]
            expr_tf = expr[:, tf_idx]
            corr = pearson_correlation(expr_tf[:, None], expr_tgs).ravel()
            tf_tg_corr += [corr.tolist()]

    # Flatten list
    if flat:
        tf_tg_corr = [c for corr_l in tf_tg_corr for c in corr_l]
        tg_tg_corr = [c for corr_l in tg_tg_corr for c in corr_l]

    return tf_tg_corr, tg_tg_corr

