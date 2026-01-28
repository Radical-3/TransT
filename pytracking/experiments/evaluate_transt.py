from pytracking.evaluation import Tracker, get_dataset, trackerlist


def evaluate_transt_got10k():
    """Evaluate TransT tracker on GOT-10k validation set"""
    # Create tracker for TransT
    trackers = trackerlist('transt', 'transt50', None)
    
    # Get GOT-10k validation dataset
    dataset = get_dataset('got10k_val')
    return trackers, dataset


def evaluate_transt_lasot():
    """Evaluate TransT tracker on LaSOT dataset"""
    # Create tracker for TransT
    trackers = trackerlist('transt', 'transt50', None)
    
    # Get LaSOT dataset
    dataset = get_dataset('lasot')
    return trackers, dataset


def evaluate_transt_otb():
    """Evaluate TransT tracker on OTB dataset"""
    # Create tracker for TransT
    trackers = trackerlist('transt', 'transt50', None)
    
    # Get OTB dataset
    dataset = get_dataset('otb')
    return trackers, dataset
