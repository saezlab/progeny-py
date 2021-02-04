import pickle
import pandas as pd
import numpy as np
import pkg_resources




def getModel(organism = "Human", top=100):
    # Get package path
    if organism == "Human":
        path = pkg_resources.resource_filename(__name__, 'data/model_human_full.pkl')
    elif organism == "Mouse":
        path = pkg_resources.resource_filename(__name__, 'data/model_mouse_full.pkl')
    else:
        raise("Wrong organism name. Please specify 'Human' or 'Mouse'.")
    
    full_model = pickle.load(open(path, "rb" ))
    
    if type(top) != int:
        raise("perm should be an integer value")
        
    model = full_model.sort_values(['pathway', 'p.value'])
    model = model.groupby('pathway').head(top)
    model = model.pivot_table(index='gene', columns='pathway', values='weight', fill_value=0)
    
    return model


