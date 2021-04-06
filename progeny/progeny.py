import pandas as pd
import numpy as np

import scanpy as sc
from anndata import AnnData

import pkg_resources
import pickle



def extract(data, key='progeny'):
    """
    Extracts values stored in `.obsm` and creates a new AnnData object.
    
    Params
    ------
    data:
        AnnData object with `.obsm` keys.
    key:
        `.obsm` key to extract.

    Returns
    -------
    Returns a new AnnData object.
    """
    
    # Get values stored in .obsm by key
    df = data.obsm[key]
    
    # Get metadata
    obsm = data.obsm
    obs = data.obs
    var = pd.DataFrame(index=df.columns)
    
    # Create new object with X as the obsm
    tadata = AnnData(np.array(df), obs=obs, var=var, obsm=obsm)
    
    return tadata

    
def getModel(organism = "Human", top=1000):
    """
    Gets gene weights for each pathway from the human progeny model (Schubert 2018) 
    or the mouse (Holland 2019) model.
    
    Params
    ------
    organism:
        Organism to use. Gene weights are only available for Human and Mouse.
    top:
        Number of top significant genes in the progeny model to use.

    Returns
    -------
    Returns dataframe with gene weights for each pathway.
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
        raise ValueError("top should be an integer value")
       
    # Select top n genes per pathway by lowest p values
    model = full_model.sort_values(['pathway', 'p.value'])
    model = model.groupby('pathway').head(top)
    model = model.pivot_table(index='gene', columns='pathway', values='weight', fill_value=0)
    
    return model


def process_input(data, use_raw=False):
    if isinstance(data, AnnData):
        if not use_raw:
            genes = np.array(data.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples = data.obs.index
            X = data.X[:,idx]
        else:
            genes = np.array(data.raw.var.index)
            idx = np.argsort(genes)
            genes = genes[idx]
            samples= data.raw.obs_names
            X = data.raw.X[:,idx]
    elif isinstance(data, pd.DataFrame):
        genes = np.array(df.columns)
        idx = np.argsort(genes)
        genes = genes[idx]
        samples = df.index
        X = np.array(df)[:,idx]
    else:
        raise ValueError('Input must be AnnData or pandas DataFrame.')
    return genes, samples, X


def run(data, model, center=True, scale=True, inplace=True, use_raw=False):
    """
    Computes pathway activity based on transcription data using progeny 
    (Schubert 2018) gene weights.
    
    Params
    ------

    Returns
    -------
    Returns pathway activities for each sample.
    """
    # Get genes, samples/pathways and matrices from data and regnet
    x_genes, x_samples, X = process_input(data, use_raw=use_raw)
    
    assert len(x_genes) == len(set(x_genes)), 'Gene names are not unique'

    if X.shape[0] <= 1 and (center or scale):
        raise ValueError('If there is only one observation no centering nor scaling can be performed.')

    # Sort targets (rows) alphabetically
    model = model.sort_index()
    m_genes, m_path = model.index, model.columns
    
    assert len(m_genes) == len(set(m_genes)), 'model gene names are not unique'
    assert len(m_path) == len(set(m_path)), 'model pathway names are not unique'

    # Subset by common genes
    common_genes = np.sort(list(set(m_genes) & set(x_genes)))
    
        
    target_fraction = len(common_genes) / len(m_genes)
    assert target_fraction > .05, \
    f'Too few ({len(common_genes)}) genes found. Make sure you are using the correct organism.'

    print(f'{len(common_genes)} genes found')
    
    idx_x = np.searchsorted(x_genes, common_genes)
    X = X[:,idx_x]
    M = model.loc[common_genes].values

    if center:
        X = X - np.mean(X, axis=0)

    # Run matrix mult
    result = np.asarray(X.dot(M))

    if scale:
        std = np.std(result, ddof=1, axis=0)
        std[std == 0] = 1
        result = (result - np.mean(result, axis=0)) / std

    # Remove nans
    result[np.isnan(result)] = 0

    # Store in df
    result = pd.DataFrame(result, columns=m_path, index=x_samples)
    
    if isinstance(data, AnnData) and inplace:
        # Update AnnData object
        data.obsm['progeny'] = result
    else:
        # Return dataframe object
        data = result

    return data if not inplace else None