import sys
sys.path.append('/home/theospira/notebooks/projects/WW_climatology/functions')

import xarray as xr
import numpy as np
from tqdm.notebook import tqdm_notebook as tqdm
from smoothing_and_interp import *
import gsw # for conversion functions

def ww_mask(ds2):
    """create data array such that contains 1 if there is winter water and 0 if no winter water
    present in pressure x grid cell
    """
    
    ww_msk = np.zeros(ds2.ctemp.shape)
    up_bd  = (ds2.up_bd / 2).astype(int).data # divide by 2 to get as indeces
    lw_bd  = (ds2.lw_bd / 2).astype(int).data # divide by 2 to get as indeces
    condn  = ds2.ctemp.notnull().sum('pres').data # grid cell contains non-nan ct data
    
    # assign
    for i in tqdm(range(360),"calc ww_mask"):
        for j in range(40):
            for k in range(4):
                if condn[k,i,j] != 0:
                    ww_msk[k,up_bd[k,i,j]:lw_bd[k,i,j],i,j] = 1
                    
    ds2['ww_msk'] = xr.DataArray(ww_msk,
                                  dims   = {'season':ds2.season.data,'pres':ds2.pres.data,
                                            'lon':ds2.lon.data,'lat':ds2.lat.data},
                                  coords = {'season':ds2.season.data,'pres':ds2.pres.data,
                                            'lon':ds2.lon.data,'lat':ds2.lat.data},
                                )
    
    return ds2

def seasonal_grouping(ds):
    """group a dataset on monthly grouping in order of winter, spring, summer, autumn seasons."""
    
    ds['month'] = [7,8,9,10,11,12,1,2,3,4,5,6]
    return ds.groupby_bins(group='month',bins=range(0,15,3),labels=range(0,4))

from warnings import filterwarnings
def load_data(ds_path='/home/theospira/notebooks/projects/WW_climatology/data/hydrographic_profiles/SO_1yr_clim_seasonal.nc'):
              #'/home/theospira/notebooks/projects/WW/ww_ds_seasonal_clim.nc'):
    """
    load, smooth and nan interpolate xr datasets, as well as bathymetry, altimetry and sea ice. 
    outputs:
    -------
    ds: unsmoothed, un-interped dataset with WW vars, sea ice, and relevant ssh
    ds_s: smoothed, nan-interped dataset with WW vars, sea ice and relevant ssh
    ds_ww: wrap_smoothed, nan-interped dataset with only WW vars
    bth: bathymetry elevation dataset
    ssh: altimetry dataset
    si: sea ice
    szn: list of season names; used in plotting :)
    
    
    """

    filterwarnings('ignore')
    
    szn = ['Winter','Spring','Summer','Autumn']
    
    ds = xr.open_dataset(ds_path)
    if 'time' in list(ds.dims.keys()):
        ds = ds.rename({'time':'season'})

    # convert depth to +ve
    #ds[['thcc','ww_cd','up_bd','lw_bd']] = ds[['thcc','ww_cd','up_bd','lw_bd']]*-1
    
    # load ssh (adt) data
    ssh = xr.open_dataset('/home/theospira/notebooks/data/Copernicus/ssh_monthly_climatology_2004-2021.nc')
    ssh = seasonal_grouping(ssh).mean('month').rename({'month_bins':'season'})

    # re-grid onto 1°x1°
    gs = 1 # grid cell size
    ssh = (ssh.coarsen(lat=int(240*gs/60),).mean(
            ).coarsen(lon=int(1440*gs/360),).mean()).sel(lat=slice(-80,-40))
    ssh['eke'] = np.sqrt(ssh.ugos**2 + ssh.vgos**2)
    
    # add adt to ds
    ds['adt'] = xr.DataArray(ssh.adt.data, dims = ('season','lat','lon'))
    
    # load bathymetry data
    bth = xr.open_mfdataset('/home/theospira/notebooks/data/GEBCO/gebco**.nc')
    
    # add bathym data to dataset
    gs = 1 # gridcell size
    bth2 = bth.sel(lat=slice(-80,-40)
            ).coarsen(lat=int(14400/60*gs),).mean(
            ).coarsen(lon=int(86400/360*gs),).mean().load()

    ds['bth'] = xr.DataArray(bth2.elevation.data,dims=('lat','lon'))
    del(bth2)
    
    # re-grid bathym data to 0.5°
    gs = 0.5 # gridcell size
    bth = bth.coarsen(lat=int(14400/60*gs),).mean(
            ).coarsen(lon=int(86400/360*gs),).mean().load()
    
    # load sea ice data and re-grid
    si = xr.open_dataset('/home/theospira/notebooks/data/meereisportal_sea_ice/sic_1yr_clim_monthly.nc')

    gs = 0.5 # grid cell size
    si = (si.coarsen(lat=int(251*gs/25),boundary='trim').mean(
            ).coarsen(lon=int(3601*gs/360),boundary='trim').mean())

    si['month'] = [7,8,9,10,11,12,1,2,3,4,5,6] # relabel into desired order, starting at 1 in Winter
    si = si.groupby_bins(group='month',bins=range(0,15,3),labels=[0,1,2,3]).mean('month').rename(
                {'month_bins':'season'})
    si = si.sortby('season')
    
    # add sic to ds
    gs = 0.25
    sic = np.ndarray(ds.adt.shape) * np.nan
    sic[:,4:29,:] = si.coarsen(lat=int(251*gs/25),boundary='trim'
                                        ).mean().coarsen(lon=int(3601*gs/360),boundary='trim').mean(
                                        ).sic.transpose('season','lat','lon').data
    ds['sic'] = xr.DataArray(sic,
                             dims = ('season','lat','lon'))
    
    # remove surface mixing layer that is highly variable 
    ds = ds.sel(pres=slice(10,300))
    
    # remove any data that is "on ground" or above sea level
    ds = ds.where(ds.bth < 0)
    
    ww_vars = ['ww_cd','ww_ct','up_bd','lw_bd','ww_n2','mld','thcc','ww_sa','sig_c','ww_type']
    # select WW data that is south of SAF 
    ds[ww_vars] = ds[ww_vars].where(ds.lat<=(ds.adt - -0.1).__abs__().idxmin(dim='lat'))
    
    ############ removing some weird profiles in Weddell Gyre. deciding to leave in for time being...
    
    # remove all WW profiles in selected grid cells -- regions of whack temperature profiles (mainly Weddell Sea, winter)
    # only removing from WW-related variables. can probs be removed from all vars? also including conservative temp
#    lon=[[111, 142, 170, 174, 175, 178, 180, 265],[174, 176, 183, 186, 187, 263]]
 #   lat=[[20, 25, 18, 17, 17, 17, 19, 22],[18, 18, 20, 20, 21, 18]]
  #  for v in ['ww_cp','ww_ct','up_bd','lw_bd','ww_n2','mlp','thcc','ww_sa','ctemp']:
   #     if v != 'ctemp':
    #        for i in range(2):
     #           for j in range(len(lon[i])):
      #              ds[v][i,lon[i][j],lat[i][j]] = np.nan
#        else:
 #           for i in range(2):
  #              for j in range(len(lon[i])):
   #                 ds[v][i,:,lon[i][j],lat[i][j]] = np.nan
    
    ############
    

    # create only WW related dataset from data variables
    # smooth that dataset using, making sure it includes data either side of cylcic point (wrap smoothing)
    # then nan interp for max gap = 3
    # cut at smoothed latitudinal boundary of WW extent
    ds_ww = ds[ww_vars].copy()
    for i in list(ds_ww.data_vars.keys()):
        if i == 'ww_type':
            continue
        ds_ww[i] = wrap_smth_var(ds_ww[i])
        ds_ww[i] = nan_interp(ds_ww[i])
        ds_ww[i] = ww_ext_bounding(ds_ww[i],ds)   

    # smooth entire dataset
    ds_s = smth_var(ds.copy())
    
#    ds_s = ds.copy()
 #   for i in list(ds.data_vars.keys()):
  #      ds_s[i] = wrap_smth_var(ds[i])
    
    # interpolate nans in WW data
    ds_s[ww_vars] = nan_interp(ds_s[ww_vars])
    
    return ds, ds_s, ds_ww, bth, ssh, si, szn