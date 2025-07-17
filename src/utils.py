"""
Helper functions, e.g., for timing functions
"""
import time


# inspiration from https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timeit(f):
    """
    # TODO: docstring
    """
    
    def timed(*args, **kwargs):
        
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        
        print("func:%r took: %2.4f sec" % \
            (f.__name__, te-ts))
        return result
    
    return timed
