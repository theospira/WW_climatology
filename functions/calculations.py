import xarray as xr
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm

def calc_mlp(ds, density_variable='rho', den_lim=0.03):
    """
    This function, `calc_mlp`, calculates the Mixed Layer Pressure (MLP) for a given dataset `ds`.
    The MLP is determined based on the density change criterion as defined by de Boyer Montegut.

    The function follows these steps:
    1. Identify the first non-null data point per profile in the dataset.
    2. Ensure that the reference depth is at least 10 dbar, or the first data point if it is less than or equal to 30 meters.
    3. Remove profiles that do not have data within the top 30 dbar.
    4. Recalculate the first non-null data point per profile after the removal of invalid profiles.
    5. Define the density change limit (den_lim) to identify the mixed layer.
    6. Calculate the Mixed Layer Pressure (mlp) by determining the pressure at which the density change exceeds the defined limit.

    Parameters:
    - ds: xarray.Dataset
        The input dataset containing oceanographic profiles with density and pressure data.
    - density_variable: str, default 'rho'
        The name of the variable representing density in the dataset.
    - den_lim: float, default 0.03
        The density change limit to identify the mixed layer.

    Returns:
    - xarray.Dataset
        The input dataset with an additional variable `mlp` representing the Mixed Layer Pressure for each profile.
    """
    # first determine first data point per profile
    ref_dpt_idx = ds[density_variable].isnull().argmin('pres') 
    # if ref depth is less than 10dbar, make ref depth 10dbar,
    # ie, ref_dept = min(10dbar, first data point ≤ 30m)
    ref_dpt_idx = ref_dpt_idx.where(ref_dpt_idx > 5, 5)
    # remove profiles without data in the top 30dbar
    ds = ds.where(ref_dpt_idx <= 15, drop=True)
    ref_dpt_idx = ds[density_variable].isnull().argmin('pres') 

    # calc mixed layer pressure
    # +5 to accommodate indexing from 5th bin onwards
    # *2 to convert from index to dbar
    ds['mlp'] = ((np.abs(ds[density_variable].isel(pres=ref_dpt_idx) - ds[density_variable].isel(pres=slice(5, 2000))) > den_lim
                  ).argmax(dim='pres', skipna=True) + 5) * 2
    
    return ds

def calc_ww(ds):
    """
    calculate winter water per profile. calculates two types of WW: surface (ML; type 1) and subsurface (SS; type 2).
    
    params:
    ------
    ds = xr.Dataset with dims=(n_prof, pres) containing data_vars: 
                ctemp (conservative temperature),
                asal (absolute salanitiy),
                mlp (mixed layer pressure) and n2 (Brunt Vaisalla frequency).
            
    returns: ds
    dataset with added variables of upper boundary, core pressure, core temperature, 
    core salintity, lower boundary, WW type, thickness, WW cumulative N2.
    """
    up_bd   = np.ndarray(ds.n_prof.size)*np.nan
    ww_cp   = np.ndarray(ds.n_prof.size)*np.nan
    ww_ct   = np.ndarray(ds.n_prof.size)*np.nan
    sig_c   = np.ndarray(ds.n_prof.size)*np.nan # core density
    ww_sa   = np.ndarray(ds.n_prof.size)*np.nan
    lw_bd   = np.ndarray(ds.n_prof.size)*np.nan
    # classification of WW: surface/mixed layer (ML) or subsurface (SS), 1 and 2 respectively
    ww_type = np.ndarray(ds.n_prof.size)*np.nan
    thcc    = np.ndarray(ds.n_prof.size)*np.nan
    ww_n2   = np.ndarray(ds.n_prof.size)*np.nan

    ww_vars = [up_bd,ww_cp,ww_ct,ww_sa,sig_c,lw_bd,ww_type,thcc,ww_n2]
    
    # create empty list of latitudes. for computation of depth values
    lat = []

    for i in tqdm(range(ds.n_prof.size),'profile idx'):
        tmp = ds.isel(n_prof=i)
        
        # use mean mixed layer temp as a reference temperature
        T_ref = tmp.ctemp.sel(pres=slice(0,tmp.mlp.data)).mean(skipna=True)

        # check if there exist WW below ML:
        cond = (tmp.ctemp.sel(pres=slice(tmp.mlp.data,400),) - T_ref < 0).sum(skipna=True)
        
        # type 2 WW (subsurf)
        if cond > 10: # then WW does exist. using more than 5 so it's not a few spurious -ves
            # if there is no data, skip this profile 
            if tmp.ctemp.sel(pres=slice(tmp.mlp.data,400)).notnull().sum('pres') < 1: 
                continue
            up_bd[i]   = tmp.mlp.data      # mixed layer
            lw_bd[i]   = tmp.ctemp.diff('pres').idxmax(skipna=True).data # point of max temp gradient

            tmp2 = tmp.sel(pres=slice(up_bd[i],lw_bd[i]))
            if tmp2.ctemp.notnull().sum('pres')!=0:              # if data exists within boundaries
                ww_cp[i]   = tmp2.ctemp.idxmin(skipna=True).data # point of minimum temperature below ML
                ww_ct[i]   = tmp2.ctemp.min(skipna=True).data    # minimum temp below ML
                ww_sa[i]   = tmp.asal.sel(pres=ww_cp[i]).data    # salanity at min temp below ML
                sig_c[i]   = tmp.sig.sel(pres=ww_cp[i]).data     # sigma (density) at min temp below ML
                ww_type[i] = 2                                   # type 2!
                
                # adjust upper bound if it has a temperature greater than 2°C to closest point 
                # to 2°C between present upper bound and core depth
                if tmp.sel(pres=up_bd[i]).ctemp > 2:
                    up_bd[i] = (tmp.sel(pres=slice(up_bd[i],ww_cp[i])).ctemp - 2).__abs__().idxmin().data
                
                # similarly, adjust lower bound if it has a temperature greater than 2°C to closest point 
                # to 2°C between present core depth and lower bound
                if tmp.sel(pres=lw_bd[i]).ctemp > 2:
                    lw_bd[i] = (tmp.sel(pres=slice(ww_cp[i],lw_bd[i])).ctemp - 2).__abs__().idxmin().data

                # calc WW vars that are based on upper and lower bounds:
                thcc[i]  = lw_bd[i]-up_bd[i] # thickness = diff in boundaries
                ww_n2[i] = tmp.n2.sel(pres=slice(up_bd[i],lw_bd[i])).sum('pres').data

                # if the mean ML temperature is greater than core temperature, no WW_SS exists
                if T_ref > ww_ct[i]:
                    for v in ww_vars:
                        v[i] = np.nan
            
            else: # else no WW exists here
                for v in ww_vars:
                    v[i] = np.nan
                
            
        # type 1 WW (ML)
        else:
            tmp2 = tmp[['ctemp','asal','sig']].sel(pres=slice(0,tmp.mlp.data))

            up_bd[i]   = 10
            ww_cp[i]   = tmp.mlp.data / 2
            ww_ct[i]   = tmp2.ctemp.mean(skipna=True).data
            ww_sa[i]   = tmp2.asal.mean(skipna=True).data
            sig_c[i]   = tmp2.sig.mean(skipna=True).data
            lw_bd[i]   = tmp.mlp
            ww_type[i] = 1
            thcc[i]    = lw_bd[i]-up_bd[i]
            ww_n2[i]   = tmp.n2.sel(pres=slice(up_bd[i],lw_bd[i])).sum('pres').data
        
        # check if this only holds for both or only type 2 WW
        if ww_ct[i] > 2: # then no WW exists in this profile
            for v in ww_vars:
                v[i] = np.nan                                 
                               
    # add variables to dataset
    ds['up_bd'] = xr.DataArray(up_bd,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['ww_cp'] = xr.DataArray(ww_cp,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['ww_ct'] = xr.DataArray(ww_ct,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['ww_sa'] = xr.DataArray(ww_sa,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['sig_c'] = xr.DataArray(sig_c,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['lw_bd'] = xr.DataArray(lw_bd,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    
    ds['ww_type'] = xr.DataArray(ww_type,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                 attrs  = {'description':'classification of WW: surface (ML) or subsurface, 1 and 2 respectively'},
                                )
    ds['thcc'] = xr.DataArray(thcc,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    ds['ww_n2'] = xr.DataArray(ww_n2,
                                 dims   = {'n_prof':ds.n_prof.data},
                                 coords = {'n_prof':ds.n_prof.data},
                                )
    
    return ds

# Import necessary libraries and modules
from scipy.ndimage import gaussian_filter1d as gf  # Import the Gaussian filter function from SciPy
import xarray as xr  # Import xarray for working with labeled multi-dimensional arrays

# Define a function for Gaussian filtering of vertical profiles in a dataset
def gauss_filter_z(ds, dvars=['temp', 'psal'], std=4):
    """
    Apply Gaussian filtering to vertical profiles of selected variables in a dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing vertical profiles of oceanographic variables.

    dvars : list of str, optional
        A list of variable names to which Gaussian filtering will be applied.
        Default is ['temp', 'psal'].

    std : int or float, optional
        The standard deviation of the Gaussian filter. Default is 4.

    Returns:
    --------
    ds : xarray.Dataset
        The modified dataset with Gaussian-filtered variables.
    """
    # Iterate through the specified variables for Gaussian filtering
    for d in dvars:
        # Apply Gaussian filtering with the specified standard deviation
        ds[d] = xr.DataArray(
            gf(ds[d].data, std),  # Gaussian filter with the specified standard deviation
            dims={'n_prof': ds.n_prof.data, 'pres': ds.pres.data},  # Preserve dimensions
            coords={'n_prof': ds.n_prof.data, 'pres': ds.pres.data}  # Preserve coordinates
        )
        
    return ds  # Return the modified dataset with Gaussian-filtered variables