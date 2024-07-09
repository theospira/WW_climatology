import xarray as xr
import numpy as np
import pandas as pd

def smth_var(dvar,x='lon',y='lat',nx=3,ny=3,mp=None,lon_wrap=False,x_wrap=20):
    """
    linearly interpolate nans; x-y coords averaged across nx-ny rolling mean; remove interpolated grid cells.
    
    params:
    -------
    dvar : input variable (DataArray or DataSet)
    x    = 'lon'
         : x dimension from dvar grid
    y    = 'lat'
         : y dimension from dvar grid
    nx   = 3
         : rolling window size in x direction
    ny   = 3
         : rolling window size in y direction
    mp   = None
         : min_periods of rolling function
    lon_wrap = True
             : whether to wrap data across x parameter (lon) (-180,180) so it is cyclical
    x_wrap = 20
           : total number of x grid cells to include on data wrapping
    """
    
    # nan interpolate
    var = dvar.interpolate_na(dim=x,method='linear').interpolate_na(dim=y,method='linear')
    if lon_wrap == True:
        var = pd.concat([var[-(x_wrap):], var, var[:(x_wrap)]]).rolling({x:nx,y:ny},min_periods=mp).mean(skipna=True)[(x_wrap):-(x_wrap)].where(dvar.notnull())
    else:
        var = var.rolling({x:nx,y:ny},min_periods=mp).mean(skipna=True).where(dvar.notnull())
    return var

def nan_interp(dvar,mg=3):
    """interpolate over small gaps of nans in x and y direction. Optionally also in z direction (if 'pres' included as a dimension).
    params:
    -------
    dvar : input variable (DataArray or DataSet)
    mg   = 3
         : max_gap as per xarray/pd interp functions: 
             "Maximum size of gap, a continuous sequence of NaNs, that will be filled."
    """
    
    if 'pres' in dvar.dims:
        dvar = dvar.interpolate_na(dim='lat',max_gap=mg).interpolate_na(dim='lon',max_gap=mg).interpolate_na(dim='pres')
    else:
        dvar = dvar.interpolate_na(dim='lat',max_gap=mg).interpolate_na(dim='lon',max_gap=mg)
    return dvar

def vert_smth(dvar,n_lat=3,n_lon=3,n_pres=10,x=None):
    """linearly interpolate nans; lon-lat n degree rolling mean; remove interpolated grid cells."""
    var = dvar.interpolate_na(dim='pres',method='linear')
    return var.rolling({'lat':n_lat,'lat':n_lon,'pres':n_pres},min_periods=x
                          ).mean(skipna=True).where(dvar.notnull())

def wrap_smth_var(dvar,x_wrap=20,x='lon',y='lat',t='season',nx=3,ny=3,nz=10,mp=None):
    """wrap smoothing of variable to go over circumpolar boundaries zonally by x_wrap (20° default)."""

    if len(dvar.dims) == 1:
        var = dvar.interpolate_na(dim=x,method='linear')
         # add on the final x_wrap number of lon data to beginning and start x_wrap data to end
        var = xr.DataArray(np.concatenate([var.data[-x_wrap:],var.data,var.data[x_wrap:]],),dims=x,)
        var = var.rolling({x:nx},min_periods=mp).mean(skipna=True).isel(
            # remove the wrapped datapoints and remove interped data
                lon=slice(x_wrap,dvar.lon.size+x_wrap)).where(dvar.notnull()) 
        
    elif len(dvar.dims) == 2:
        dvar = dvar.transpose(x, y)
        var = dvar.interpolate_na(dim=x,method='linear').interpolate_na(dim=y,method='linear')
         # add on the final x_wrap number of lon data to beginning and start x_wrap data to end
        var = xr.DataArray(np.concatenate([var.data[-x_wrap:,:],var.data,var.data[x_wrap:,:]],
                                          axis=0),dims=(x,y),)
        var = var.rolling({x:nx,y:ny},min_periods=mp).mean(skipna=True).isel(
            # remove the wrapped datapoints and remove interped data
                lon=slice(x_wrap,dvar.lon.size+x_wrap)).where(dvar.notnull()) 
        
    elif len(dvar.dims) == 3:
        dvar = dvar.transpose(t, x, y)
        var = dvar.interpolate_na(dim=x,method='linear').interpolate_na(dim=y,method='linear')
        var = xr.DataArray(np.concatenate(
            # add on the final x_wrap number of lon data to 
            [var.data[:,-x_wrap:,:],var.data,var.data[:,x_wrap:,:]], 
              axis=1), # beginning and start x_wrap data to end
              dims=(t,x,y),)
        var = var.rolling({x:nx,y:ny},min_periods=mp).mean(skipna=True).isel(
            # remove the wrapped datapoints and remove interped data
                lon=slice(x_wrap,dvar.lon.size+x_wrap)).where(dvar.notnull()) 
        
    elif len(dvar.dims) == 4:
        var = dvar.interpolate_na(dim=x,method='linear').interpolate_na(dim=y,method='linear'
                    ).interpolate_na(dim='pres',method='linear')
        # add on the final x_wrap number of lon data to beginning and start x_wrap data to end
        var = xr.DataArray(np.concatenate([dvar.data[:,:,-x_wrap:,:],dvar.data,dvar.data[:,:,x_wrap:,:]],axis=2),dims=(t,'pres',x,y),)
        var = var.rolling({x:nx,y:ny,'pres':nz},min_periods=mp).mean(skipna=True).isel(
                # remove the wrapped datapoints and remove interped data
                lon=slice(x_wrap,dvar.lon.size+x_wrap)).where(dvar.notnull()) 
    return var

# Define a function to calculate the bounding latitude for WW profiles, selecting only south of the northern-most extent
def ww_ext_bounding(var, ds_og):
    """
    Calculate the bounding latitude for specific oceanographic profiles in a dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The input dataset containing oceanographic data profiles.

    var : xr.DataArray
        The variable name to calculate the bounding latitude for.

    ds_og : xarray.Dataset, optional
        The original dataset containing the WW (Winter Water) type variable. Default is ds.

    Returns:
    --------
    ds : xarray.Dataset
        The modified dataset with calculated bounding latitudes.
    """
    # Extract the WW (Winter Water) type variable
    ww = ds_og['ww_type']
    ww = ww.where(np.logical_or(ww == 1, ww == 2))
    
    # Initialize an empty list to store results for each season
    tmp = []

    # Iterate over four seasons
    for i in range(4):
        # Calculate the bounding latitude for the current season
        dvar = ww.notnull().cumsum('lat').idxmax(dim='lat', skipna=True)[i]

        # Set latitude values further north than 45°S or south than 70°S to NaN
        dvar[dvar < -70] = np.nan
        dvar[dvar > -45] = np.nan

        # Add data to start and end for wrapping .diff()
        dvar2 = xr.DataArray(np.concatenate([dvar[-20:], dvar, dvar[:20]]), dims='lon')
        dvar2 = dvar2.interpolate_na(dim='lon', method='cubic')

        # Find locations where there are significant differences
        idx = np.where(dvar2.diff('lon').__abs__() > 4)[0] + 1
        dvar2[idx] = np.nan

        # Store the resulting bounding latitude with appropriate lon values
        tmp += np.floor(dvar2.interpolate_na(dim='lon', method='cubic'))[20:-20],
        tmp[i]['lon'] = dvar.lon

        # Update the variable in the original dataset where lat is less than the computed latitude
        var[i] = var[i].where(var[i].lat < wrap_smth_var(tmp[i], 20, nx=5))

    return var