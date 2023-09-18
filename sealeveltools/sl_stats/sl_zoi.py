def zoi_function(x,kind='cor',how='rel',level=20.,merge='mean'):
    """
    select subsets based on criteria, kinds and levels (abs./relative)

    criteria:
        - standard
        - oc_fraction
        - mission_lvls
        - absolute
        - distribution
        - cut_depth
        ...
    kinds
        - rms
        - corr
        - AC

    """
    if how=='rel':
        if kind=='cor':
            x=x.where(x >= x.quantile(level))
        elif kind=='rms':
            x=x.where(x <= x.quantile(1-level))
        elif kind=='AC':        
            x=x.where(x <= x.quantile(1-level))
    #if merge =='':
    return x
    #else:
    #    return getattr(x, merge)()