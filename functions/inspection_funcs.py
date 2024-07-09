import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs # for plotting

def boxplot(var,showfliers=False,ax=None,**kwargs):
    """variable input is a ds.var format (xr DataArray)"""
    if ax==None:
        ax = plt.gca()
    var = var.data.flatten()
    ax.boxplot(var[np.isfinite(var)],showfliers=False,**kwargs)
    
def no_nans(var):
    return var[np.isfinite(var)]