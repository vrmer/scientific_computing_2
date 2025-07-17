"""
Helper functions, e.g., for timing functions
"""
from dataclasses import fields


# from ChatGPT
def filter_to_dataclass(dataclass_type, data: dict):
    """
    Given a dataclass and a dictionary with config parameters,
    it fills in the required parameters of the dataclass with 
    the ones provided through the config file.
    
    Parameters:
        dataclass_type: a dataclass
        data: parameters from a config file (dict)
        
    Output:
        an initialised dataclass of the type 'dataclass_type'
        
    The function was co-written with ChatGPT.
    """
    valid_keys = {f.name for f in fields(dataclass_type)}
    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
    return dataclass_type(**filtered_data)
