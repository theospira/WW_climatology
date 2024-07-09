import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs # for plotting
import cartopy.feature as cfeature # for map features

def circular_boundary(ax):
    """
    Create a circular boundary for a map plot.

    This function computes a circular boundary in axes coordinates, which can be used as a boundary
    for a map plot. It allows for panning and zooming, ensuring that the boundary remains circular.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The axis to which the circular boundary will be applied.

    Returns:
    --------
    None

    Notes:
    ------
    The circular boundary is set using the `set_boundary` method of the provided axis.

    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    >>> circular_boundary(ax)
    >>> ax.set_theta_zero_location('N')
    >>> ax.set_theta_direction(-1)
    >>> ax.set_rmax(1)
    >>> plt.show()
    """
    # Generate theta values for a complete circle
    theta = np.linspace(0, 2 * np.pi, 100)

    # Define the center and radius of the circle in axes coordinates
    center, radius = [0.5, 0.5], 0.5

    # Calculate the vertices of the circular path
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T

    # Create a matplotlib Path object representing the circular boundary
    circle = mpath.Path(verts * radius + center)

    # Set the circular boundary for the specified axis
    ax.set_boundary(circle, transform=ax.transAxes)


def plot_nice_box(ax,x1,x2,y1,y2,**kwargs):
    """
    Plot a box on a SouthPolar Stereo cartopy figure with curved edges.
    
    Parameters:
    -----------
    ax = axis
    
    x1 = lon1
    x2 = lon2
    y1 = lat1
    y2 = lat2
    
    """
    
    ax.plot([x1,x1],[y1,y2],transform=ccrs.PlateCarree(),**kwargs) # Straight vertical line
    ax.plot([x2,x2],[y1,y2],transform=ccrs.PlateCarree(),**kwargs) # Straight vertical line
    ax.plot(np.linspace(x1,x2,1000),[y2]*1000,transform=ccrs.PlateCarree(),**kwargs) # Nice curve
    ax.plot(np.linspace(x1,x2,1000),[y1]*1000,transform=ccrs.PlateCarree(),**kwargs)
    
    
# function to plot fronts on each axis
def plot_fronts(ax,i,lon,n,fronts):
    """
    ---
    Params:
    ax  : axis
    i   : index
    lon : list of longitude sections to plot
    n   : index of lons
    fronts : fronts xarray dataset
    """
    # locate PF
    idx = (fronts.LonPF - lon[n]).__abs__().argmin()
    x   = fronts.LatPF[idx]
    ax[i].axvline(x=x,c='k',lw=3,ls='--')
    ax[i].annotate('PF',(x+0.5,980),fontsize=30,color='k',zorder=10,weight='bold')
    # locate SAF
    idx = (fronts.LonSAF - n).__abs__().argmin()
    x   = fronts.LatSAF[idx]
    ax[i].axvline(x=x,c='k',lw=3,ls='--')
    ax[i].annotate('SAF',(x+0.5,980),fontsize=30,color='k',zorder=10,weight='bold')

    ax[i].axes.invert_yaxis()
    ax[i].set_xlabel('')
    ax[i].set_ylabel('')
    
import matplotlib as mpl
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
def gridlines(ax,lon_tick,lat_tick):
    """add gridlines to an axis. lon_tick and y_tick are gridsize and input as real numbers."""
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      x_inline=False, y_inline=False,
                      linewidth=0.75, alpha=0.5, linestyle='--',
                      ylocs = mpl.ticker.MultipleLocator(base=lat_tick),
                      xlocs = mpl.ticker.MultipleLocator(base=lon_tick))

    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LongitudeFormatter
    gl.yformatter = LatitudeFormatter

    gl.xpadding=10
    gl.ypadding=10
    
# Function: update_projection

def update_projection(ax, axi, projection, fig=None):
    """
    Update a subplot's projection.

    Parameters:
    -----------
    ax : matplotlib.axes._subplots.AxesSubplot
        The main axis from the whole figure containing subplots.

    axi : matplotlib.axes._subplots.AxesSubplot
        The subplot axis to re-project.

    projection : cartopy.crs.Projection
        The desired new projection for the subplot.

    fig : matplotlib.figure.Figure, optional
        The figure to which the subplot belongs. If not provided, it is retrieved using plt.gcf().

    Description:
    ------------
    This function updates the projection of a subplot within a multi-subplot figure.
    It removes the existing subplot with its current projection and replaces it with a new
    subplot using the specified projection.

    Notes:
    ------
    - You can check available projections using matplotlib.projections.get_projection_names()
      or use Cartopy options for projections.

    Example:
    --------
    # Usage example
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    # Create a multi-subplot figure
    fig, ax = plt.subplots(2, 2, figsize=(8, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Update the projection of a specific subplot
    update_projection(ax, ax[1, 1], ccrs.Mercator())
    plt.show()
    """
    if fig is None:
        fig = plt.gcf()

    # Get the geometry of the subplot's position in the grid
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()

    # Remove the existing subplot
    ax.flat[start].remove()

    # Add a new subplot with the desired projection
    ax.flat[start] = fig.add_subplot(rows, cols, start + 1, projection=projection)



from cartopy.util import add_cyclic_point
def wrap_data_func(data,):
    """
    Wrap Data Along Longitude Axis
    
    This function adds a cyclic point to input data along the longitude axis.
    It is useful for visualizations to prevent artifacts at edges with discontinuous longitude coordinates.
    
    Parameters:
        data (xarray.DataArray or xarray.Dataset): Input data with a 'lon' coordinate.
    
    Returns:
        wrap_data (numpy.ndarray): Wrapped data with an added cyclic point.
        wrap_lon (numpy.ndarray): Modified longitude coordinates with the cyclic point.
    
    Usage:
    1. Import required libraries: xarray and add_cyclic_point from cartopy.util.
    2. Call the function: wrap_data, wrap_lon = wrap_data(your_data).
    3. Utilize the wrapped data and longitude coordinates in visualizations or analyses.
    
    Example:
        ds = xr.open_dataset('your_data.nc')
        your_data = ds['your_variable']
        wrap_data, wrap_lon = wrap_data(your_data)
    """
    wrap_data, wrap_lon = add_cyclic_point(data.values, coord=data.coords['lon'], axis=data.dims.index('lon'))
    return wrap_data, wrap_lon

from smoothing_and_interp import wrap_smth_var,nan_interp
class DatasetError(Exception):
    pass

def circular_plot_fomatting(fig,ax,ds,si=[],set_fc=True,fc='darkgrey',
                            fronts=True,bth=[],bathym=True,sea_ice=True,
                            annotation=True,si_x=15,a_x=0.06,a_y=0.89,legend=False,
                            leg_loc=(0.8725,0.91),extent=[180,-180,-90,-45]):
    """
    Apply formatting to subplots with circular boundaries for South Polar Stereo projections.
    
    This function formats subplots within a given figure to create South Polar Stereo projections
    with circular boundaries. It adds map features such as land, coastlines, fronts, bathymetry,
    and sea ice contours. Annotations, legends, and other visual elements can be customized.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to which the subplots belong.
        
    ax : matplotlib.axes._subplots.AxesSubplot or numpy.ndarray of them
        The subplot(s) to be formatted. If a numpy array of subplots is provided, the formatting
        will be applied to each subplot.
        
    ds : xarray.Dataset
        A dataset containing oceanographic observations gridded to 1° by 1° in lat-lon space
        in climatological time (seasons). It should include an 'adt' variable.
        
    si : list of xarray.Dataset, optional
        A list of sea ice datasets. Default is an empty list. Ensure it is filled if 'sea_ice' is True.
        
    set_fc : bool, optional
        If True, set the background face color of the subplots. Default is True.
        
    fc : str, optional
        The background face color of the subplots. Default is 'darkgrey'.
        
    fronts : bool, optional
        If True, plot fronts on the subplots. Default is True.
        
    bth : list of xarray.Dataset, optional
        A list of bathymetry datasets with an 'elevation' variable. Default is an empty list.
        Ensure it is filled if 'bathym' is True.
        
    bathym : bool, optional
        If True, plot 1, 2, and 3 km bathymetry contours on the subplots. Default is True.
        
    sea_ice : bool, optional
        If True, plot sea ice concentration contours on the subplots. Default is True.
        
    si_x : float, optional
        The sea ice concentration threshold for plotting contours. Default is 15%.
        
    a_x, a_y : float, optional
        The x and y coordinates in subplot coordinate space for subplot annotations. Default is (0.06, 0.89).
        
    annotation : bool, optional
        If True, add subplot label annotations (alphabetically). Default is True.
        
    legend : bool, optional
        If True, add a legend to the figure. Default is False.
        
    leg_loc : tuple, optional
        The legend location in figure (x, y) coordinates. Default is (0.8725, 0.91).
        
    extent : list of float, optional
        The projection extent in the format [E, W, S, N] (East, West, South, North).
        Default is [180, -180, -90, -45].
    """
    
    alphbt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    crs = ccrs.PlateCarree()
    
    if type(ax)==np.ndarray:
        for i,a in enumerate(ax):
            leg_arr = [] # legend handles to be included
            leg_lab = [] # legend labels  to be included
            circular_boundary(a)
            a.set_extent(extent,crs=crs)
            a.add_feature(cfeature.LAND,zorder=9,facecolor="#ededed",)
            a.coastlines(zorder=10,color="k") # shows continental coastline

            if set_fc:
                a.set_facecolor(fc)

            # add fronts, bathym and sea ice contours
            if fronts:
                wrap_data, wrap_lon = wrap_data_func(ds.adt.mean('season'))
                frnt = a.contour(wrap_lon,ds.lat.data,wrap_data,levels=[-0.58,-0.1],colors='k',
                          linewidths=1.5,linestyles=['-'],labels='PF & SAF',transform=crs,)
                #frnt = ds.adt.mean('season').plot.contour(x='lon',transform=crs,ax=a,levels=[-0.58,-0.1],colors='k',linewidths=1.5,linestyles=['-'],labels='PF & SAF')
                leg_arr += frnt, 
                leg_lab += "PF,SAF",

            if bathym:
                if len(bth)==0:
                    raise DatasetError("define xarray bathymetry dataset with an elevation variable")

                wrap_data, wrap_lon = wrap_data_func(bth.elevation)
                b = a.contour(wrap_lon,bth.lat.data,wrap_data,transform=crs,levels=[-3e3,-2e3,-1e3],
                          colors='#4c4c4c',linestyles=['-'],linewidths=[0.7],alphas=[0.9])
                leg_arr += b,
                leg_lab += "1,2,3km isobaths",

            if sea_ice:
                if len(si)==0:
                    raise DatasetError("define xarray sea ice (si) dataset with a sea ice concentration (sic) variable")

                ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').isel(season=int(i%4)),nx=20
                           ).plot(ax=a,c='w',transform=crs,lw=1.25)
                leg_arr += ice[0],
                leg_lab += f"sic ({si_x}%)",

            # add annotation for each subfig
            if annotation:
                a.text(x=a_x,y=a_y,s="(" + alphbt[i] + ")",transform=a.transAxes)
    
    elif len(ax)>1:
        for i,a in enumerate(ax):
            leg_arr = [] # legend handles to be included
            leg_lab = [] # legend labels  to be included
            circular_boundary(a)
            a.set_extent(extent,crs=crs)
            a.add_feature(cfeature.LAND,zorder=9,facecolor="#ededed",)
            a.coastlines(zorder=10,color="k") # shows continental coastline

            if set_fc:
                a.set_facecolor(fc)

            # add fronts, bathym and sea ice contours
            if fronts:
                wrap_data, wrap_lon = wrap_data_func(ds.adt.mean('season'))
                frnt = a.contour(wrap_lon,ds.lat.data,wrap_data,levels=[-0.58,-0.1],colors='k',
                                 linewidths=1.5,linestyles=['-'],labels='PF & SAF',transform=crs,)
                leg_arr += frnt, 
                leg_lab += "PF,SAF",

            if bathym:
                if len(bth)==0:
                    raise DatasetError("define xarray bathymetry dataset with an elevation variable")

                wrap_data, wrap_lon = wrap_data_func(bth.elevation)
                b = a.contour(wrap_lon,bth.lat.data,wrap_data,transform=crs,levels=[-3e3,-2e3,-1e3],
                          colors='#4c4c4c',linestyles=['-'],linewidths=[0.7],alphas=[0.9])
                leg_arr += b,
                leg_lab += "1,2,3km isobaths",

            if sea_ice:
                if len(si)==0:
                    raise DatasetError("define xarray sea ice (si) dataset with a sea ice concentration (sic) variable")

                ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').isel(season=int(i%4)),nx=20
                           ).plot(ax=a,c='w',transform=crs,lw=1.25)
                leg_arr += ice[0],
                leg_lab += f"sic ({si_x}%)",

            # add annotation for each subfig
            if annotation:
                a.text(x=a_x,y=a_y,s="(" + alphbt[i] + ")",transform=a.transAxes)
    
    # add legend
    if legend:
        fig.legend(handles=leg_arr,labels=leg_lab,loc=leg_loc,facecolor='#c3c3c3')
    
def circular_plot_fomatting_single_ax(fig,ax,ds,si=[],set_fc=True,fc='darkgrey',
                            fronts=True,bth=[],bathym=True,sea_ice=True,
                            annotation=True,si_x=15,a_x=0.06,a_y=0.89,legend=False,
                            leg_loc=(0.8725,0.91),extent=[180,-180,-90,-45]):
    crs = ccrs.PlateCarree()
    a   = ax
    leg_arr = [] # legend handles to be included
    leg_lab = [] # legend labels  to be included
    circular_boundary(a)
    a.set_extent(extent,crs=crs)
    a.add_feature(cfeature.LAND,zorder=9,facecolor="#ededed",)
    a.coastlines(zorder=10,color="k") # shows continental coastline

    if set_fc:
        a.set_facecolor(fc)

    # add fronts, bathym and sea ice contours
    if fronts:
        wrap_data, wrap_lon = wrap_data_func(ds.adt.mean('season'))
        frnt = a.contour(wrap_lon,ds.lat.data,wrap_data,levels=[-0.58,-0.1],colors='k',
                  linewidths=1.5,linestyles=['-'],labels='PF & SAF',transform=crs,)
        leg_arr += frnt.collections[0], 
        leg_lab += "PF,SAF",

    if bathym:
        if len(bth)==0:
            raise DatasetError("define xarray bathymetry dataset with an elevation variable")
        wrap_data, wrap_lon = wrap_data_func(bth.elevation)
        b = a.contour(wrap_lon,bth.lat.data,wrap_data,transform=crs,levels=[-3e3,-2e3,-1e3],
                  colors='#4c4c4c',linestyles=['-'],linewidths=[0.7],alphas=[0.9])
        leg_arr += b[0],
        leg_lab += "1,2,3km isobaths",

    if sea_ice:
        if len(si)==0:
            raise DatasetError("define xarray sea ice (si) dataset with a sea ice concentration (sic) variable")

        ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').isel(season=int(i%4)),nx=20
                   ).plot(ax=a,c='w',transform=crs,lw=1.25)
        leg_arr += ice[0],
        leg_lab += f"sic ({si_x}%)",

    # add legend
    if legend:
        fig.legend(handles=leg_arr,labels=leg_lab,loc=leg_loc,facecolor='#c3c3c3')
        
# Define a function to create a figure displaying seasonal differences of a variable
def plot_seasonal_diff(var, ax, **kwargs):
    """
    Create a figure displaying seasonal differences of a variable.

    Parameters:
    -----------
    var : xarray.DataArray
        The input variable with seasonal data.

    ax : list of matplotlib.Axes
        A list of subplot axes where the seasonal differences will be plotted.

    kwargs : keyword arguments
        Additional keyword arguments to customize the plots.

    Returns:
    --------
    hmp : xarray.DataArray
        The result of the plot function.
    """
    crs = ccrs.PlateCarree()

    for i, a in enumerate(ax):
        hmp = (var.isel(season=i) - var.isel(season=(int(i - 1) % 4))).plot(x='lon', transform=crs, ax=a, **kwargs)

    return hmp

# Define a function to create a figure displaying the mean seasonal change of a variable
def plot_mean_seasonal_change(fig, ax, var, var_lab, units, si, cmap, vmin, vmax, szn_vmax, szn_cmap, shrink=0.7, asp=12, fs=20,
                              set_ticks=False, ticks=[], set_ticks2=False, ticks2=[], si_x=15, invert_cb=False):
    """
    Create a figure displaying the mean seasonal change of a variable and sea ice contours.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to which the subplots belong.
        
    ax : list of matplotlib.Axes
        A list of subplot axes where the plots will be displayed.

    var : xarray.DataArray
        The input variable with seasonal data.

    var_lab : str
        The label for the variable.

    units : str
        The units of the variable.

    si : xarray.DataArray
        Sea ice data.

    si_x : int, optional
        Sea ice parameter.

    cmap : str
        The colormap for the plot.

    vmin : float
        The minimum value for the color scale.

    vmax : float
        The maximum value for the color scale.

    szn_vmax : float
        The maximum value for the seasonal change plot.

    szn_cmap : str
        The colormap for the seasonal change plot.

    shrink : float, optional
        The shrink factor for the colorbars.

    asp : int, optional
        The aspect ratio of the colorbars.

    fs : int, optional
        The fontsize for labels and colorbar labels.

    set_ticks : bool, optional
        Whether to set colorbar ticks.

    ticks : list, optional
        Custom colorbar ticks.

    set_ticks2 : bool, optional
        Whether to set colorbar ticks for the seasonal change plot.

    ticks2 : list, optional
        Custom colorbar ticks for the seasonal change plot.

    Returns:
    --------
    None
    """
    crs = ccrs.PlateCarree()

    # Plot the mean
    hmp = var.mean('season').plot(x='lon', transform=crs, ax=ax[0], add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
    cb = fig.colorbar(hmp, ax=ax[0], shrink=shrink, aspect=asp, location='left')
    cb.set_label(label=var_lab + '\n(' + units + ')', fontsize=fs)
    if set_ticks:
        cb.set_ticks(ticks)
    if invert_cb:
        cb.ax.invert_yaxis()

    # Plot the seasonal change
    hmp = plot_seasonal_diff(var, ax[1:5], vmin=-szn_vmax, vmax=szn_vmax, cmap=szn_cmap, add_colorbar=False)
    cb = fig.colorbar(hmp, ax=ax[1:5], shrink=shrink, aspect=asp)
    cb.set_label(label='Seasonal\nChange (' + units + ')', fontsize=fs)
    if set_ticks2:
        cb.set_ticks(ticks2)
        
    # Plot sea ice contour
    ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').mean('season'), nx=20
                       ).plot(ax=ax[0], c='w', transform=crs, lw=1.25)
    for i, a in enumerate(ax[1:5]):
        ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').isel(season=i), nx=20
                           ).plot(ax=a, c='w', transform=crs, lw=1.25)


# Define a function to create a figure displaying seasonal anomalies of a variable
def plot_seasonal_anomaly(var, ax, **kwargs):
    """
    Create a figure displaying seasonal anomalies of a variable.

    Parameters:
    -----------
    var : xarray.DataArray
        The input variable with seasonal data.

    ax : list of matplotlib.Axes
        A list of subplot axes where the seasonal anomalies will be plotted.

    kwargs : keyword arguments
        Additional keyword arguments to customize the plots.

    Returns:
    --------
    hmp : xarray.DataArray
        The result of the plot function.
    """
    crs = ccrs.PlateCarree()

    for i, a in enumerate(ax):
        hmp = (var.isel(season=i) - var.mean('season')).plot(x='lon', transform=crs, ax=a, **kwargs)

    return hmp

# Define a function to create a figure displaying the mean anomaly of a variable and sea ice contours
def plot_mean_anomaly(fig, ax, var, var_lab, units, si, cmap, vmin, vmax, szn_vmax, szn_cmap, shrink=0.7, asp=12, fs=20,
                      set_ticks=False, ticks=[], set_ticks2=False, ticks2=[], si_x=15, invert_cb=False):
    """
    Create a figure displaying the mean anomaly of a variable and sea ice contours.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to which the subplots belong.
        
    ax : list of matplotlib.Axes
        A list of subplot axes where the plots will be displayed.

    var : xarray.DataArray
        The input variable with seasonal data.

    var_lab : str
        The label for the variable.

    units : str
        The units of the variable.

    si : xarray.DataArray
        Sea ice data.

    cmap : str
        The colormap for the plot.

    vmin : float
        The minimum value for the color scale.

    vmax : float
        The maximum value for the color scale.

    szn_vmax : float
        The maximum value for the seasonal anomaly plot.

    szn_cmap : str
        The colormap for the seasonal anomaly plot.

    shrink : float, optional
        The shrink factor for the colorbars.

    asp : int, optional
        The aspect ratio of the colorbars.

    fs : int, optional
        The fontsize for labels and colorbar labels.

    set_ticks : bool, optional
        Whether to set colorbar ticks.

    ticks : list, optional
        Custom colorbar ticks.

    set_ticks2 : bool, optional
        Whether to set colorbar ticks for the seasonal anomaly plot.

    ticks2 : list, optional
        Custom colorbar ticks for the seasonal anomaly plot.

    si_x : int, optional
        Sea ice parameter.

    Returns:
    --------
    None
    """
    crs = ccrs.PlateCarree()

    # Plot the mean
    hmp = var.mean('season').plot(x='lon', transform=crs, ax=ax[0], add_colorbar=False, vmin=vmin, vmax=vmax, cmap=cmap)
    cb = fig.colorbar(hmp, ax=ax[0], shrink=shrink, aspect=asp, location='left')
    cb.set_label(label=var_lab + '\n(' + units + ')', fontsize=fs)
    if set_ticks:
        cb.set_ticks(ticks)
    if invert_cb:
        cb.ax.invert_yaxis()

    # Plot the anomaly
    hmp = plot_seasonal_anomaly(var, ax[1:5], vmin=-szn_vmax, vmax=szn_vmax, cmap=szn_cmap, add_colorbar=False)
    cb = fig.colorbar(hmp, ax=ax[1:5], shrink=shrink, aspect=asp)
    cb.set_label(label='Anomaly\n(' + units + ')', fontsize=fs)
    if set_ticks2:
        cb.set_ticks(ticks2)

    # Plot sea ice contour
    ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').mean('season'), nx=20
                       ).plot(ax=ax[0], c='w', transform=crs, lw=1.25)
    for i, a in enumerate(ax[1:5]):
        ice = wrap_smth_var((si.sic - si_x).__abs__().idxmin(dim='lat').isel(season=i), nx=20
                           ).plot(ax=a, c='w', transform=crs, lw=1.25)


def set_share_axes(axs, target=None, sharex=False, sharey=False):
    """
    Set shared x or y axes for a list of subplots.

    Parameters:
        axs (numpy.ndarray): 2D array of subplots.
        target (matplotlib.axes._subplots.AxesSubplot, optional): The target subplot for sharing axes.
            If not provided, the first subplot (axs.flat[0]) is used as the target.
        sharex (bool): Whether to share x-axis.
        sharey (bool): Whether to share y-axis.

    Returns:
        None

    Note:
        The provided axs should be a 2D array representing the subplots.
    """
    if target is None:
        target = axs.flat[0]

    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_axes['x'].join(target, ax)
        if sharey:
            target._shared_axes['y'].join(target, ax)

    # Turn off x tick labels and offset text for all but the bottom row
    if sharex and axs.ndim > 1:
        for ax in axs[:-1, :].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)

    # Turn off y tick labels and offset text for all but the left-most column
    if sharey and axs.ndim > 1:
        for ax in axs[:, 1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)

def plot_bathym(ax,bth):
    """
    plot bathymetry with its features labelled on a subplot axis.
    """

    bth.elevation.plot.contour(x='lon',levels=[-1e3,-2e3,-3e3,],linestyles=['-'],colors='k',
                               linewidths=[1],alphas=[0.9],ax=ax,zorder=2)
    hmp = (bth.elevation*-1).plot(x='lon',cmap='Blues',ax=ax,zorder=1,vmin=0,vmax=5000,#levels=31,
                                  add_colorbar=False,)
#                            cbar_kwargs={'extend':'max','label':'Depth (m)',}).colorbar
#    cb.ax.invert_yaxis()
#    cb.set_ticks(np.arange(0,5050,1000))
#    cb.minorticks_off()
    cm = mpl.colors.LinearSegmentedColormap.from_list("", ['#ededed','#ededed'])
    bth.elevation.plot.contour(x='lon',levels=[0],colors='k',linestyles=['-'],linewidths=[1.5],ax=ax)
    bth.elevation.where(bth.elevation>0).plot(x='lon',cmap=cm,ax=ax,add_colorbar=False,)
    
    
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_ylim(-80,-45)
    ax.set_xticks(np.arange(-180,190,60))
    ax.set_yticks(np.arange(-80,-42.5,5))
    
    txt_kwrgs=dict(fontsize=20,c='#f5f5f5',weight='bold',zorder=3)
    ax.text(x=-165,y=-71,s='RS',  **txt_kwrgs)
    ax.text(x=-152,y=-60,s='PAR', **txt_kwrgs)
    ax.text(x=-106,y=-67,s='ABS', **txt_kwrgs)
    ax.text(x=-70, y=-60,s='DP',  **txt_kwrgs)
    ax.text(x=-30, y=-66.5,s='WS',**txt_kwrgs)
    ax.text(x=-35, y=-55,s='SSA', **txt_kwrgs)
   # ax.text(x=-0.5,y=-58,s='MAR', **txt_kwrgs)
    ax.text(x=12,  y=-52,s='SWIR',**txt_kwrgs)
    ax.text(x=80,  y=-56.5,s='KP',**txt_kwrgs)
    ax.text(x=119, y=-60,s='SEIR',**txt_kwrgs)
    return hmp

def add_bathym_vlines(ax,l_kwargs={'c':'grey','ls':'--'}):
    """
    add vertical lines to subplot axis of bathymetry features.
    includes PAR, SSA, SWIR, KP, SEIR
    """
    
    ax.axes.axvline(-145,**l_kwargs) #PAR
    ax.axes.axvline(-30, **l_kwargs) # SSA
    ax.axes.axvline(25,  **l_kwargs) # SWIR
    ax.axes.axvline(79,  **l_kwargs) # KP
    ax.axes.axvline(145, **l_kwargs) # SEIR