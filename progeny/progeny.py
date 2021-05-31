import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
from anndata import AnnData
import pickle
import pkg_resources
import os
from numpy.random import default_rng
from tqdm import tqdm


def load_model(organism = "Human", top=None):
    """
    Gets gene weights for each pathway from the human progeny model (Schubert 2018) 
    or the mouse (Holland 2019) model.
    
    Params
    ------
    organism:
        Organism to use. Gene weights are only available for Human and Mouse.
    top:
        Number of top significant genes per pathway in the progeny model to use.

    Returns
    -------
    Returns DataFrame with gene weights for each pathway.
    """
    # Set model path
    path = 'data'
    fname = 'model_'
    
    if organism == "Human" or organism == "Mouse":
        fname += organism.lower()
    else:
        raise ValueError("Wrong organism name. Please specify 'Human' or 'Mouse'.")
        
    path = pkg_resources.resource_filename(__name__, os.path.join(path, fname + '_full.pkl'))
        
    # Load model
    full_model = pickle.load(open(path, "rb" ))
       
    # Select top n genes per pathway by lowest p values
    model = full_model.sort_values(['pathway', 'p.value'])
    if top is not None:
        model = model.groupby('pathway').head(top)
    model = model.pivot_table(index='gene', columns='pathway', values='weight', fill_value=0)
    
    return model


def extract(adata, obsm_key='progeny'):
    """
    Generates a new AnnData object with pathway activities stored in `.obsm` instead of gene expression. 
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    obsm_key
        `.osbm` key where pathway activities are stored.
    
    Returns
    -------
    AnnData object with pathway activities
    """
    obsm = adata.obsm
    obs = adata.obs
    df = adata.obsm[obsm_key]
    var = pd.DataFrame(index=df.columns)
    pw_adata = AnnData(np.array(df), obs=obs, var=var, obsm=obsm)
    return pw_adata


def process_input(data, use_raw=False, use_hvg=False):
    """
    Processes different input types so that they can be used downstream. 
    
    Parameters
    ----------
    data
        Annotated data matrix or DataFrame
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    use_hvg
        If data is an AnnData object, whether to only use high variable genes.
    
    Returns
    -------
    genes : list of genes names
    samples : list of sample names
    X : gene expression matrix
    """
    if isinstance(data, AnnData):
        if not use_raw:
            genes = np.array(data.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples = data.obs.index
            X = data.X[:,idx]
            if use_hvg:
                hvg_msk = data.var.loc[genes].highly_variable
                X = X[:,hvg_msk]
                genes = genes[hvg_msk]
        else:
            genes = np.array(data.raw.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples= data.raw.obs_names
            X = data.raw.X[:,idx]
            if use_hvg:
                hvg_msk = data.raw.var.loc[genes].highly_variable
                X = X[:,hvg_msk]
                genes = genes[hvg_msk]
    elif isinstance(data, pd.DataFrame):
        genes = np.array(data.columns)
        idx = np.argsort(genes)
        genes = genes[idx]
        samples = data.index
        X = np.array(data)[:,idx]
    else:
        raise ValueError('Input must be AnnData or pandas DataFrame.')
    return genes, samples, csr_matrix(X)


def dot_mult(X, M):
    # Run matrix mult
    pw_act = np.asarray(X.dot(M))
    return pw_act


def scale_arr(X, scale_axis):
    std = np.std(X, ddof=1, axis=scale_axis)
    std[std == 0] = 1
    mean = np.mean(X, axis=scale_axis)
    if scale_axis == 0:
        X = (X - mean) / std
    elif scale_axis == 1:
            X = (X - mean.reshape(-1,1)) / std.reshape(-1,1)
    return X


def center_arr(X):
    X = X.copy()
    sums = np.squeeze(X.sum(1).A)
    counts = np.diff(X.tocsr().indptr)
    means = sums/counts
    X.data -= np.repeat(means, counts)
    return X


def run(data, model, center=True, num_perm=0, norm=True, scale=True, scale_axis=0, inplace=True, 
        use_raw=False, use_hvg=False, obsm_key='progeny', min_size=5):
    """
    Computes pathway activity based on transcription data using progeny 
    (Schubert 2018) gene weights.
    
    Parameters
    ----------
    data
        Annotated data matrix or DataFrame.
    model
        PROGENy model in DataFrame format.
    center
        Whether to center gene expression by cell/sample.
    num_perm
        Number of permutations to calculate p-vals of random activities.
    norm
        Whether to normalize activities per regulon size to correct for large regulons.
    scale
        Whether to scale the final activities.
    scale_axis
        0 to scale per feature, 1 to scale per cell/sample.
    inplace
        If `data` is an AnnData object, whether to update `data` or return a DataFrame.
    use_raw
        If data is an AnnData object, whether to use values stored in `.raw`.
    use_hvg
        If data is an AnnData object, whether to only use high variable genes.
    obsm_key
        `.osbm` key where pathway activities will be stored.
    min_size
        Pathways with regulons with less targets than `min_size` will be ignored.
    
    Returns
    -------
    Returns a DataFrame with pathway activities or adds it to the `.obsm` key 'dorothea' 
    of the input AnnData object, depending on `inplace` and input data type.
    """
    
    # Get genes, samples/pathways and matrices from data and regnet
    x_genes, x_samples, X = process_input(data, use_raw=use_raw, use_hvg=use_hvg)
    
    assert len(x_genes) == len(set(x_genes)), 'Gene names are not unique'

    if center:
        X = center_arr(X)

    # Sort targets (rows) alphabetically
    model = model.sort_index()
    m_genes, m_path = model.index, model.columns
    
    assert len(m_genes) == len(set(m_genes)), 'model gene names are not unique'
    assert len(m_path) == len(set(m_path)), 'model pathway names are not unique'

    # Subset by common genes
    common_genes = np.sort(list(set(m_genes) & set(x_genes)))
    
    target_fraction = len(common_genes) / len(m_genes)
    assert target_fraction > .05, f'Too few ({len(common_genes)}) genes found. \
    Make sure you are using the correct organism.'

    print(f'{len(common_genes)} genes found')
    
    idx_x = np.searchsorted(x_genes, common_genes)
    X = X[:,idx_x]
    M = model.loc[common_genes].values
        
    # Check min size and filter
    msk_size = np.sum(M != 0, axis=0) < min_size
    num_small_reg = np.sum(msk_size)
    if num_small_reg > 0:
        print(f'{num_small_reg} Pathways with < {min_size} targets')
        M[:, msk_size] = 0

    # Run matrix mult
    estimate = dot_mult(X, M)
    
    # Permutations
    if num_perm > 0:
        pvals = np.zeros(estimate.shape)
        for i in tqdm(range(num_perm)):
            perm = dot_mult(X, default_rng(seed=i).permutation(M))
            pvals += np.abs(perm) > np.abs(estimate)
        pvals = pvals / num_perm
        pvals[pvals == 0] = 1/num_perm
    else:
        pvals = np.full(estimate.shape, 0.1)
        
    # Normalize by num edges
    if norm:
        norm = np.sum(np.abs(M), axis=0)
        norm[norm == 0] = 1
        estimate = estimate / norm
        
    # Weight estimate by pvals
    pw_act = estimate * -np.log10(pvals)
    
    # Scale output
    if scale:
        pw_act = scale_arr(pw_act, scale_axis)
    
    # Store in df
    result = pd.DataFrame(pw_act, columns=m_path, index=x_samples)
    
    if isinstance(data, AnnData) and inplace:
        # Update AnnData object
        data.obsm[obsm_key] = result
    else:
        # Return dataframe object
        data = result
        inplace = False

    return data if not inplace else None


def rank_pws_groups(adata, groupby, group, reference='all'):
    """
    Runs Wilcoxon rank-sum test between one group and a reference group.
    
    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    group
        Group or list of groups to compare.
    reference
        Reference group or list of reference groups to use as reference.
    
    Returns
    -------
    DataFrame with changes in pathway activity between groups.
    """
    from scipy.stats import ranksums
    from statsmodels.stats.multitest import multipletests

    # Get pathway activites
    adata = extract(adata)
    
    # Get pathway names
    features = adata.var.index.values

    # Generate mask for group samples
    if isinstance(group, str):
        g_msk = (adata.obs[groupby] == group).values
    else:
        cond_lst = [(adata.obs[groupby] == grp).values for grp in group]
        g_msk = np.sum(cond_lst, axis=0).astype(bool)
        group = ', '.join(group)

    # Generate mask for reference samples
    if reference == 'all':
        ref_msk = ~g_msk
    elif isinstance(reference, str):
        ref_msk = (adata.obs[groupby] == reference).values
    else:
        cond_lst = [(adata.obs[groupby] == ref).values for ref in reference]
        ref_msk = np.sum(cond_lst, axis=0).astype(bool)
        reference = ', '.join(reference)
        
    assert np.sum(g_msk) > 0, 'No group samples found'
    assert np.sum(ref_msk) > 0, 'No reference samples found'

    # Wilcoxon rank-sum test 
    results = []
    for i in np.arange(len(features)):
        stat, pval = ranksums(adata.X[g_msk,i], adata.X[ref_msk,i])
        mc = np.mean(adata.X[g_msk,i]) - np.mean(adata.X[ref_msk,i])
        results.append([features[i], group, reference, stat, mc, pval])

    # Tranform to df
    results = pd.DataFrame(
        results, 
        columns=['name', 'group', 'reference', 'statistic', 'meanchange', 'pval']
    ).set_index('name')
    
    # Correct pvalues by FDR
    results[np.isnan(results['pval'])] = 1
    _, pvals_adj, _, _ = multipletests(
        results['pval'].values, alpha=0.05, method='fdr_bh'
    )
    results['pval_adj'] = pvals_adj
    
    # Sort by statistic
    results = results.sort_values('meanchange', ascending=False)
    return results

