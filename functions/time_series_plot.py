# Import necessary libraries
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime as dt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define a function to create a profile timeseries plot
def profile_timeseries_sp(ax,freq='3M'):
    """
    Create a profile timeseries plot for hydrographic data.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis where the profile timeseries plot will be created.

    Description:
    ------------
    This function loads hydrographic profile data from a NetCDF file, groups the data into 3-month intervals,
    and counts the number of profiles for each data source within each interval. It then creates a stacked bar chart
    to visualize the profile counts over time for different data sources.

    The function takes a Matplotlib axis (`ax`) as input and adds the profile timeseries plot to it.

    Notes:
    ------
    - Make sure to provide a valid path to the NetCDF file containing the hydrographic profile data.
    - This function uses several external libraries, including xarray, pandas, numpy, and tqdm.
    - Customize data sources and associated colors in the `dsrce` and `c` lists as needed.
    - Adjust the x-axis limits as desired.
    """
    # Load hydrographic profile data from a NetCDF file (update the file path)
    tmp = xr.open_dataset('/home/theospira/notebooks/projects/WW_climatology/data/hydrographic_profiles/ww_gauss_smoothed_ds.nc')

    # Create time bins for grouping the data in 3-month intervals

    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq=freq)

    # Group the data by time into 3-month intervals and count profiles in each interval
    grp = tmp[['n_prof', 'dsource']].groupby_bins(tmp.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    tmp3 = grp.count('n_prof')
    tmp3 = tmp3.rename({'time_bins': 'time'})
    del(tmp)

    # Initialize an array for plotting
    arr = np.ndarray([5, tmp3.time.size]) * np.nan

    # Define data sources and associated colors
    dsrce = ['Argo',    'MEOP',    'CTD',     'Gliders', 'SOCCOM']
    c     = ['#9e0142', '#86cfa5', '#f98e52', '#ffffbe', '#5e4fa2']

    # Loop through time bins and data sources to count profiles
    for j, t in enumerate(list(grp.groups.keys())):
        for i, d in enumerate(dsrce):
            dsr = grp[t].dsource
            if d in dsr:
                arr[i, j] = (dsr == d).sum()

    # Initialize the bottom of the bars for stacking
    bottom = np.zeros(tmp3.time.size)

    # Create a bar chart for each data source with stacked bars
    if freq=='3M':
        x_ax = t_bins[:-1] + pd.DateOffset(months=1) + pd.DateOffset(days=14)
        w    = 75
    elif freq=='1M':
        x_ax = t_bins[:-1] + pd.DateOffset(days=14)
        w    = 25
    for i, d in enumerate(dsrce):
        ax.bar(x_ax, arr[i], label=d, bottom=bottom, color=c[i], width=w,
               edgecolor=None, linewidth=0)
        arr2 = arr[i]
        arr2[np.isnan(arr[i])] = 0
        bottom += arr2

    # Set the x-axis limits
    ax.set_xlim(t_bins[0], dt.strptime('2022-01-01', '%Y-%M-%d'))

def total_dsr_pcnt(ax):
    dsrce = ['Argo',    'MEOP',   'SOCCOM',  'CTD',     'Gliders', ]
    c     = ['#9e0142', '#86cfa5','#5e4fa2', '#f98e52', '#ffffbe', ]

    tmp = xr.open_dataset('/home/theospira/notebooks/projects/WW_climatology/data/hydrographic_profiles/ww_gauss_smoothed_ds.nc')
    # what are the total number of profiles for each data source
    dsr = np.unique(tmp.dsource,return_counts=True)
    # as percentage
#    arr = np.round(dsr[1] / dsr[1].sum() * 100,2)
    arr = dsr[1] / dsr[1].sum() * 100


    bottom = 0
    for i, d in enumerate(dsrce):
        idx = np.where(dsr[0] == d)
        ax.bar(1, arr[idx], label=d, bottom=bottom, color=c[i], #width=w,
               edgecolor=None, linewidth=0)
        bottom += arr[idx]

# Import necessary libraries
from datetime import datetime as dt

# Define a function to create a profile timeseries plot
def ww_profile_timeseries_sp(ax,freq='3M'):
    """
    Create a profile timeseries plot for hydrographic data.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis where the profile timeseries plot will be created.

    Description:
    ------------
    This function loads hydrographic profile data from a NetCDF file, groups the data into 3-month intervals,
    and counts the number of profiles for each data source within each interval. It then creates a stacked bar chart
    to visualize the profile counts over time for different data sources.

    The function takes a Matplotlib axis (`ax`) as input and adds the profile timeseries plot to it.

    Notes:
    ------
    - Make sure to provide a valid path to the NetCDF file containing the hydrographic profile data.
    - This function uses several external libraries, including xarray, pandas, numpy, and tqdm.
    - Customize data sources and associated colors in the `dsrce` and `c` lists as needed.
    - Adjust the x-axis limits as desired.
    """
    # Load hydrographic profile data from a NetCDF file (update the file path)
    tmp = xr.open_dataset('/home/theospira/notebooks/projects/WW_climatology/data/hydrographic_profiles/superseded/ww_gauss_smoothed_ds-preDec23.nc')

    # Create time bins for grouping the data in 3-month intervals

    t_bins = pd.date_range('2003-12-31', '2022-01-01', freq=freq)

    # Group the data by time into 3-month intervals and count profiles in each interval
    grp = tmp[['n_prof', 'ww_type']].groupby_bins(tmp.time, bins=t_bins, labels=np.arange(len(t_bins)-1))
    tmp3 = grp.count('n_prof')
    tmp3 = tmp3.rename({'time_bins': 'time'})
    del(tmp)

    # Initialize an array for plotting
    arr = np.ndarray([2, tmp3.time.size]) * np.nan

    # Define data sources and associated colors
    ww_type = ['WW$_{ML}$','WW$_{SS}$']
    c       = ['#377eb8','#ff7f00']

    # Loop through time bins and data sources to count profiles
    for j, t in enumerate(list(grp.groups.keys())):
        for i, d in enumerate(ww_type):
            dsr = grp[t].ww_type
            if i+1 in dsr:
                arr[i, j] = (dsr == int(i+1)).sum().data

    # Initialize the bottom of the bars for stacking
    bottom = np.zeros(tmp3.time.size)

    # Create a bar chart for each data source with stacked bars
    if freq=='3M':
        x_ax = t_bins[:-1] + pd.DateOffset(months=1) + pd.DateOffset(days=14)
        w    = 75
    elif freq=='1M':
        x_ax = t_bins[:-1] + pd.DateOffset(days=14)
        w    = 25
    for i, d in enumerate(ww_type):
        ax.bar(x_ax, arr[i], label=d, bottom=bottom, color=c[i], width=w,
               edgecolor=None, linewidth=0)
        arr2 = arr[i]
        arr2[np.isnan(arr[i])] = 0
        bottom += arr2

    # Set the x-axis limits
    ax.set_xlim(t_bins[0], dt.strptime('2022-01-01', '%Y-%M-%d'))

from mpl_toolkits.axes_grid1 import make_axes_locatable
def total_ww_pcnt(ax):
    ww_type = ['WW$_{ML}$','WW$_{SS}$']
    c       = ['#377eb8','#ff7f00']

    tmp = xr.open_dataset('/home/theospira/notebooks/projects/WW_climatology/data/hydrographic_profiles/ww_gauss_smoothed_ds.nc')
    # what are the total number of profiles for each data source
    ww_type = np.unique(tmp.ww_type,return_counts=True)
    # as percentage
    arr = np.round(ww_type[1] / ww_type[1].sum() * 100,2)

    bottom = 0
    for i, d in enumerate(ww_type):
        idx = np.where(dsr[0] == i+1)
        ax.bar(1, arr[idx], label=d, bottom=bottom, color=c[i], #width=w,
               edgecolor=None, linewidth=0)
        bottom += arr[idx]