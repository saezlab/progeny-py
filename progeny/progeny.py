import pickle
import pandas as pd
import numpy as np
import pkg_resources
import scanpy as sc
from anndata import AnnData


def getModel(organism = "Human", top=100):
    # Get package path
    if organism == "Human":
        path = pkg_resources.resource_filename(__name__, 'data/model_human_full.pkl')
    elif organism == "Mouse":
        path = pkg_resources.resource_filename(__name__, 'data/model_mouse_full.pkl')
    else:
        raise("Wrong organism name. Please specify 'Human' or 'Mouse'.")
        
    # Load model
    full_model = pickle.load(open(path, "rb" ))
    
    if type(top) != int:
        raise("perm should be an integer value")
       
    # Select top n genes per pathway by lowest p values
    model = full_model.sort_values(['pathway', 'p.value'])
    model = model.groupby('pathway').head(top)
    model = model.pivot_table(index='gene', columns='pathway', values='weight', fill_value=0)
    
    return model


def run(data, scale=True, organism="Human", top = 100):
    # Transform to df if AnnData object is given
    if isinstance(data, AnnData):
        if data.raw is None:
            data = pd.DataFrame(np.transpose(data.X), index=data.var.index, 
                                   columns=data.obs.index)
        else:
            data = pd.DataFrame(np.transpose(data.raw.X.toarray()), index=data.raw.var.index, 
                                   columns=data.raw.obs_names)

    # Get PROGENy model
    model = getModel(organism, top=top)
    
    # Check overlap of genes
    common_genes = np.array(model.index.intersection(data.index).to_list())
    
    # Matrix multiplication
    result = np.array(data.loc[common_genes].T.dot(model.loc[common_genes,]))
    
    if scale:
        result = (result - np.mean(result, axis=0)) / np.std(result, axis=0)

    # Return AnnData object
    pw_data = AnnData(result)
    pw_data.obs.index = data.columns
    pw_data.var.index = model.columns
    
    return pw_data