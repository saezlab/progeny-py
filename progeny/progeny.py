import pickle
import pandas as pd
import numpy as np
import pkg_resources
import scanpy as sc
from anndata import AnnData
import seaborn as sns


def plot_matrixplot(adata, groupby, cmap='coolwarm', ax=None):
    # Get progeny data
    X = adata.obsm['X_progeny']
    p_names = adata.uns['progeny']
    # Get group categroies
    cats = adata.obs[groupby].cat.categories
    # Compute mean for each group
    arr = np.zeros((len(cats),X.shape[1]))
    for i, cat in enumerate(cats):
        msk = adata.obs[groupby] == cat
        mean_group = np.mean(X[msk,], axis=0)
        arr[i] = mean_group
    # Plot heatmap
    sns.heatmap(arr, cmap=cmap, center=0, xticklabels=p_names, yticklabels=cats, robust=True, 
            square=True, cbar_kws={'shrink':0.45}, ax=ax)

    
def getModel(organism = "Human", top=100):
    """
    Gets gene weights for each pathway from the human progeny model [Schubert18]_ 
    or the mouse [Holland19]_ model.
    
    Params
    ------
    organism:
        Organism to use. Gene weights are only available for Human and Mouse.
    top:
        Number of top significant genes in the progeny model to use.

    Returns
    -------
    Returns gene weights for each pathway.
    """
    
    # Get package path
    if organism == "Human":
        path = pkg_resources.resource_filename(__name__, 'data/model_human_full.pkl')
    elif organism == "Mouse":
        path = pkg_resources.resource_filename(__name__, 'data/model_mouse_full.pkl')
    else:
        raise ValueError("Wrong organism name. Please specify 'Human' or 'Mouse'.")
        
    # Load model
    full_model = pickle.load(open(path, "rb" ))
    
    if type(top) != int:
        raise ValueError("perm should be an integer value")
       
    # Select top n genes per pathway by lowest p values
    model = full_model.sort_values(['pathway', 'p.value'])
    model = model.groupby('pathway').head(top)
    model = model.pivot_table(index='gene', columns='pathway', values='weight', fill_value=0)
    
    return model


def run(data, scale=True, organism="Human", top=100, inplace=True):
    """
    Computes pathway activity based on transcription data using progeny 
    [Schubert18]_ gene weights.
    
    Params
    ------
    data
        If `AnnData`, the annotated data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
        If `data frame`, the annotated data matrix of shape `n_vars` × `n_obs`.
    scale:
        Scale the resulting pathway activities.
    organism:
        Organism to use. Gene weights are only available for Human and Mouse.
    top:
        Number of top significant genes in the progeny model to use.
    inplace:
        Whether to update `adata` or return dictionary with normalized copies of
        `adata.X` and `adata.layers`.

    Returns
    -------
    Returns pathway activities for each sample.
    """
    
    # Transform to df if AnnData object is given
    if isinstance(data, AnnData):
        if data.raw is None:
            df = pd.DataFrame(np.transpose(data.X), index=data.var.index, 
                                   columns=data.obs.index)
        else:
            df = pd.DataFrame(np.transpose(data.raw.X.toarray()), index=data.raw.var.index, 
                                   columns=data.raw.obs_names)
    else:
        df = data

    # Get progeny model
    model = getModel(organism, top=top)
    
    # Check overlap of genes
    common_genes = np.array(model.index.intersection(df.index).to_list())
    
    # Matrix multiplication
    result = np.array(df.loc[common_genes].T.dot(model.loc[common_genes,]))
    
    if scale:
        result = (result - np.mean(result, axis=0)) / np.std(result, axis=0)

    if isinstance(data, AnnData):
        # Update AnnData object
        data.obsm['X_progeny'] = result
        data.uns['progeny'] = model.columns
    else:
        # Return dataframe object
        data = pd.DataFrame(result, columns=model.columns, index=df.columns)
    
    return data if not inplace else None