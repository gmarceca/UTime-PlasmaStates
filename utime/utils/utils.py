"""
A set of general utility functions used across the codebase
"""

import numpy as np
import logging
import os

def create_folders(folders, create_deep=False):
    def safe_make(path, make_func):
        try:
            make_func(path)
        except FileExistsError:
            # If running many jobs in parallel this may occur
            pass
    make_func = os.mkdir if not create_deep else os.makedirs
    if isinstance(folders, str):
        if not os.path.exists(folders):
            safe_make(folders, make_func)
    else:
        folders = list(folders)
        for f in folders:
            if f is None:
                continue
            if not os.path.exists(f):
                safe_make(f, make_func)

def assert_all_loaded(pairs, raise_=True):
    """
    Returns True if all SleepStudy objects in 'pairs' have the 'loaded'
    property set to True, otherwise returns False.

    If raise_ is True, raises a NotImplementedError if one or more objects are
    not loaded. Otherwise, returns the value of the assessment.

    Temp. until queue functionality implemented
    """
    loaded_pairs = [p for p in pairs if p.loaded]
    if len(loaded_pairs) != len(pairs):
        if raise_:
            raise NotImplementedError("BatchSequence currently requires all"
                                      " samples to be loaded")
        else:
            return False
    return True


def ensure_list_or_tuple(obj):
    """
    Takes some object and wraps it in a list - i.e. [obj] - unless the object
    is already a list or a tuple instance. In that case, simply returns 'obj'

    Args:
        obj: Any object

    Returns:
        [obj] if obj is not a list or tuple, else obj
    """
    return [obj] if not isinstance(obj, (list, tuple)) else obj

