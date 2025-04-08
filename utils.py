import numpy as np
from pylab import show, figure, plot
import time
from landlab import RasterModelGrid
from landlab.plot.imshow import imshow_grid
from landlab.io import read_esri_ascii
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from scipy import stats
from osgeo import gdal
import glob
# from PIL import Image
# import matplotlib.cbook as cbook
# import scipy.optimize
# from scipy.optimize import curve_fit
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
from landlab.components import TaylorNonLinearDiffuser
import imageio

### TO DO:
# Review all code
# Clean up unused packages
# Test evolution of 2 standard deviations of slopes 
# DEM lighting, constant limits for hillshade
# Make summary plots tighter 

def plot_evolution_time_linear(n_iter, DEM, shapefiles_input, epsg_code, save_YN, initial_slopes, D=0.001):
    fig, ax = plt.subplots(
    len(n_iter),4,
    tight_layout=False,
    figsize=(5.5,9),
    dpi=300)
    # set overall title
    fig.suptitle(str(DEM)) 
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.1, hspace=0.3)
    
    # to save in run 
    coeff_t = []
    years_t = []
    line_length = [] # for later plot
    
    # landlab grid from DEM
    DEM_name = 'DEMS/' + DEM + '.asc'
    mg, z = read_esri_ascii(DEM_name, name='topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    slope_t0 = mg.calc_slope_at_node(z)
    slope_t0 = np.array(slope_t0)
    z_t0 = z[mg.nodes]

    # model set-up for 2D linear diffusion
    dt = 0.2 * mg.dx * mg.dx / D # default time step is 50 years 
    qs = mg.add_zeros('sediment_flux', at='link')

    # run linear model over time
    plot_counter=0
    for p in range(max(n_iter)+1):
        if np.any(p == n_iter):
            if p*dt<dt:
                ValueError("The total time is smaller than the time-step!!")
        
            # plot hillshade
            fig.sca(ax[plot_counter,0])
            hillshade = mg.calc_hillshade_at_node(elevs=z, alt=30., az=100.)
            imshow_grid(mg,hillshade,cmap='gray',vmin=0,vmax=1) # plot_type, 'Hillshade'
            ax[plot_counter,0].set_xticklabels([])
            ax[plot_counter,0].set_yticklabels([])
            ax[plot_counter,0].set_xticks([])
            ax[plot_counter,0].set_yticks([])
            ax[plot_counter,0].set_ylabel('')
            ax[plot_counter,0].set_xlabel('')
            ax[plot_counter,0].set_aspect('equal')
            colorbar = plt.gci().colorbar
            colorbar.remove()
            
            slope_t = mg.calc_slope_at_node(z)
            
            # plot elevation difference between t and t0
            zfin = z[mg.nodes]
            z_diff =  zfin - z_t0
            zchange = mg.node_vector_to_raster(z_diff, flip_vertically=True)
            im = ax[plot_counter,2].imshow(zchange,cmap='cividis',vmin=-0.8, vmax=0.8)
            if plot_counter == 0:
                fig.colorbar(im, ax=ax[plot_counter,2],label='$\Delta$ z (m)',orientation='horizontal',fraction=0.036)
            colorbar = plt.gci().colorbar
            ax[plot_counter,2].set_yticks([])
            ax[plot_counter,2].set_xticks([])
            slope_t = np.array(slope_t)
            slope_t0 = np.array(slope_t0)
            
            # calculate degradation coefficient
            info_loss = estimate_degradation_coefficient(slope_t0,slope_t,plot_counter,ax)
            coeff_t.append(info_loss)    
            ax[plot_counter,0].set_title('t = %.0f years' %(p*dt),fontsize=6)
            years_t.append(p*dt)
            ax[plot_counter,3].set_xlabel('Slope',fontsize=6)
            ax[plot_counter,3].set_ylabel('')
            ax[plot_counter,3].set_yticks([])
            ax[plot_counter,3].set_yscale('log')  
            ax[plot_counter,3].set_xlim([0,1])  
            if plot_counter ==1:
                ax[plot_counter, 3].legend(fontsize=5)
            
            # plot shapefile of fault traces
            gdf = shapefiles_input[plot_counter]
            gdf = gdf.to_crs(epsg=epsg_code)
            if gdf.empty:
                print("The shapefile is empty.")
                print(shapefiles_input[plot_counter])
            gdf.plot(ax=ax[plot_counter, 1],linewidth=0.8, color='slategrey')
            ax[plot_counter,1].set_ylabel('')
            ax[plot_counter,1].set_yticks([])
            ax[plot_counter,1].set_xlabel('')
            ax[plot_counter,1].set_xticks([])     
            x_min, x_max = ax[plot_counter, 0].get_xlim()
            y_min, y_max = ax[plot_counter, 0].get_ylim()
            ax[plot_counter, 1].set_xlim(x_min, x_max)
            ax[plot_counter, 1].set_ylim(y_min, y_max)    
            gdf['length'] = gdf.geometry.length
            total_length = gdf.geometry.length.sum()
            ax[plot_counter,1].set_title(r"$L$ = {:.2f} m".format(total_length),fontsize=6)
            line_length.append(total_length)
            plot_counter += 1
        
        g = mg.calc_grad_at_link(z)
        qs[mg.active_links] = -D * g[mg.active_links]
        dzdt = -mg.calc_flux_div_at_node(qs)
        z[mg.core_nodes] += dzdt[mg.core_nodes] * dt  
        
    
    initial_slopes.append(slope_t0)    
    scalebar = ScaleBar(
        0.5,
        units="m",
        dimension="si-length",
        label=None,
        length_fraction=None,
        height_fraction=None,
        width_fraction=None,
        location=None,
        pad=None,
        border_pad=None,
        sep=None,
        frameon=None,
        color=None,
        box_color=None,
        box_alpha=0,
        scale_loc=None,
        label_loc=None,
        font_properties=None,
        label_formatter=None,
        scale_formatter=None,
        fixed_value=None,
        fixed_units=None,
        animated=False,
        rotation=None)

    ax[0,0].add_artist(scalebar)

    if save_YN == 'Yes':
        DEMname = str(DEM)           
        first_char = DEMname[0]
        numeric_chars = ''.join(filter(str.isdigit, DEMname))
        DEMID = first_char + numeric_chars

        txtname = 'Figures/' + DEMID + '_information_loss_analysis_linear.png'
        plt.savefig(txtname)

    return line_length, coeff_t, years_t, initial_slopes       
    
def plot_evolution_time_nonlinear(n_iter, DEM, shapefiles_input, epsg_code, save_YN, initial_slopes, D=0.001):
    fig, ax = plt.subplots(
    len(n_iter),4,
    tight_layout=False,
    figsize=(8,10),
    dpi=300)
    
    # set overall title
    fig.suptitle(str(DEM)) 

    # make variables to save from this run
    coeff_t = []
    years_t = []
    line_length = [] # for later plot
    
    # landlab grid from DEM
    DEM_name = 'DEMS/' + DEM + '.asc'
    mg, z = read_esri_ascii(DEM_name, name='topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    slope_t0 = mg.calc_slope_at_node(z)
    slope_t0 = np.array(slope_t0)
    z_t0 = z[mg.nodes]

    # model set-up for 2D non-linear diffusion
    dt = 0.2 * mg.dx * mg.dx / D # default time step is 50 years in linear model - using here for plotting reference
    cubicflux = TaylorNonLinearDiffuser(mg, linear_diffusivity=D, if_unstable="warn",dynamic_dt=True,nterms=2) #nterms from Ganti et al. (2012)
    
    # run nonlinear model over time
    plot_counter=0
    for p in range(max(n_iter)+1):
        if np.any(p == n_iter):
            # plot hillshade
            total_time = int(p * dt) # hack to get total time under loop plotting structure we built for the linear case  
            cubicflux.run_one_step(total_time)
            
            fig.sca(ax[plot_counter,0])
            hillshade = mg.calc_hillshade_at_node(elevs=z, alt=30., az=100.)
            imshow_grid(mg,hillshade,cmap='gray') # plot_type, 'Hillshade'
            ax[plot_counter,0].set_xticklabels([])
            ax[plot_counter,0].set_yticklabels([])
            ax[plot_counter,0].set_xticks([])
            ax[plot_counter,0].set_yticks([])
            ax[plot_counter,0].set_ylabel('')
            ax[plot_counter,0].set_xlabel('')
            colorbar = plt.gci().colorbar
            colorbar.remove()
            
            slope_t = mg.calc_slope_at_node(z)
            
            # plot elevation difference between t and t0
            zfin = z[mg.nodes]
            z_diff =  zfin - z_t0
            zchange = mg.node_vector_to_raster(z_diff, flip_vertically=True)
            im = ax[plot_counter,2].imshow(zchange,cmap='cividis',vmin=-0.8, vmax=0.8)
            if plot_counter == 0:
                fig.colorbar(im, ax=ax[plot_counter,2],label='$\Delta$ z (m)',orientation='horizontal')
            colorbar = plt.gci().colorbar
            ax[plot_counter,2].set_yticks([])
            ax[plot_counter,2].set_xticks([])
            slope_t = np.array(slope_t)
            slope_t0 = np.array(slope_t0)
            
            
            # calculate degradation coefficient
            info_loss = estimate_degradation_coefficient(slope_t0,slope_t,plot_counter,ax)
            coeff_t.append(info_loss)    
            ax[plot_counter,0].set_title('t = %.0f years' %(p*dt),fontsize=6)
            years_t.append(p*dt)
            ax[plot_counter,3].set_xlabel('Slope',fontsize=6)
            ax[plot_counter,3].set_ylabel('')
            ax[plot_counter,3].set_yticks([])
            ax[plot_counter,3].set_yscale('log')  
            ax[plot_counter,3].set_xlim([0,1])  
            
            # plot shapefile of fault traces
            gdf = shapefiles_input[plot_counter]
            gdf = gdf.to_crs(epsg=epsg_code)
            if gdf.empty:
                print("The shapefile is empty.")
                print(shapefiles_input[plot_counter])
            gdf.plot(ax=ax[plot_counter, 1],linewidth=0.8, color='slategrey')
            ax[plot_counter,1].set_ylabel('')
            ax[plot_counter,1].set_yticks([])
            ax[plot_counter,1].set_xlabel('')
            ax[plot_counter,1].set_xticks([])     
            ax[plot_counter,1].set_aspect('equal')   
            gdf['length'] = gdf.geometry.length
            total_length = gdf.geometry.length.sum()
            ax[plot_counter,1].set_title(r"$L$ = {:.2f} m".format(total_length),fontsize=8)
            line_length.append(total_length)
            plot_counter += 1
    
    initial_slopes.append(slope_t0)    
    scalebar = ScaleBar(
        0.5,
        units="m",
        dimension="si-length",
        label=None,
        length_fraction=None,
        height_fraction=None,
        width_fraction=None,
        location=None,
        pad=None,
        border_pad=None,
        sep=None,
        frameon=None,
        color=None,
        box_color=None,
        box_alpha=0,
        scale_loc=None,
        label_loc=None,
        font_properties=None,
        label_formatter=None,
        scale_formatter=None,
        fixed_value=None,
        fixed_units=None,
        animated=False,
        rotation=None)

    ax[0,0].add_artist(scalebar)
    plt.subplots_adjust(left=0.05, right=1, bottom=0.05, top=0.95, wspace=0.3, hspace=0.3)
    # plt.tight_layout()    
    
    if save_YN == 'Yes':
        DEMname = str(DEM)           
        first_char = DEMname[0]
        numeric_chars = ''.join(filter(str.isdigit, DEMname))
        DEMID = first_char + numeric_chars

        txtname =  'Figures/' + DEMID + '_information_loss_analysis_nonlinear.png'
        plt.savefig(txtname)

    return line_length, coeff_t, years_t, initial_slopes

def estimate_degradation_coefficient(slope_t0, slope_t, plot_counter, ax, nbins=20):
    cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    cleaned_slope_t = slope_t[~np.isnan(slope_t)]   

    percentile_threshold = 68
    threshold_t0 = np.percentile(cleaned_slope_t0, percentile_threshold)
    threshold_t = np.percentile(cleaned_slope_t, percentile_threshold)
    
    info_loss = threshold_t0/threshold_t
    
    # Handle both 1D and 2D axis arrays
    if len(ax.shape) == 1:  # 1D array (1x3 grid)
        target_ax = ax[2]  # Last axis in 1x3 grid
    else:  # 2D array (4x1 grid)
        target_ax = ax[plot_counter, 3]  # Last axis in 4x1 grid
    
    target_ax.hist(cleaned_slope_t0, color='teal', histtype='step', alpha=0.9, label='Initial slopes')
    target_ax.hist(cleaned_slope_t, color='darkorange', histtype='step', alpha=0.9, label='Slopes at time t')
    target_ax.set_title(r"$\phi$ = {:.2f}".format(info_loss), fontsize=6)
    return info_loss

def load_shapefiles_for_DEMs(DEM_dir, shapefile_dir):
    DEMs_shapefiles = {}
    
    for dem_file in os.listdir(DEM_dir):
        if dem_file.endswith('.asc'):
            dem_name = os.path.splitext(dem_file)[0]
            shapefile_subdir = os.path.join(shapefile_dir, dem_name)
            
            if os.path.isdir(shapefile_subdir):
                shapefiles_input = os.listdir(shapefile_subdir)
                shapefiles = [file for file in shapefiles_input if file.endswith(".shp")]
                shapefiles_sorted = sorted(shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0]))
                
                shapefiles_data = []
                for shp_file in shapefiles_sorted:
                    shapefile_path = os.path.join(shapefile_subdir, shp_file)
                    shapefile = gpd.read_file(shapefile_path)
                    shapefiles_data.append(shapefile)
                
                DEMs_shapefiles[dem_name] = shapefiles_data
    
    return DEMs_shapefiles

def func_deg_coeff(x,a,c):
    return x**a * c

def func_line_length(x,a,b,c):
    return b - (c * x) / (x + a)

def normalize_length(row,length_at_time_zero):
    return row['Length (m)'] / length_at_time_zero[row['DEM ID']]

def estimate_degradation_coefficient_noplot(slope_t0,slope_t):
    cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    cleaned_slope_t = slope_t[~np.isnan(slope_t)]   

    percentile_threshold = 68
    threshold_t0 = np.percentile(cleaned_slope_t0, percentile_threshold)
    threshold_t = np.percentile(cleaned_slope_t, percentile_threshold)
    
    info_loss = threshold_t0/threshold_t
    return info_loss

def create_evolution_gif(n_iter, DEM, shapefiles_input, epsg_code, D=0.001):
    # Create directory for frames if it doesn't exist
    frames_dir = 'gifs_evolution/frames'
    if not os.path.exists('gifs_evolution'):
        os.makedirs('gifs_evolution')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    
    # landlab grid from DEM
    DEM_name = 'DEMS/' + DEM + '.asc'
    mg, z = read_esri_ascii(DEM_name, name='topographic__elevation')
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    slope_t0 = mg.calc_slope_at_node(z)
    slope_t0 = np.array(slope_t0)
    z_t0 = z[mg.nodes]

    # model set-up for 2D linear diffusion
    dt = 0.2 * mg.dx * mg.dx / D  # default time step is 50 years 
    qs = mg.add_zeros('sediment_flux', at='link')

    # Set consistent figure parameters
    fig_width = 12
    fig_height = 4
    dpi = 300

    # run linear model over time
    frame_counter = 0
    for p in range(max(n_iter)+1):
        if np.any(p == n_iter):
            # Create figure for this timestep
            fig, ax = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.1)
            
            # plot hillshade
            hillshade = mg.calc_hillshade_at_node(elevs=z, alt=30., az=100.)
            plt.sca(ax[0])  # Set current axis
            imshow_grid(mg, hillshade, cmap='gray', vmin=0, vmax=1)  # Plot hillshade
            # Remove colorbar
            colorbar = plt.gci().colorbar
            if colorbar is not None:
                colorbar.remove()
            ax[0].set_xticklabels([])
            ax[0].set_yticklabels([])
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_xlabel('')
            ax[0].set_ylabel('')
            ax[0].set_title('t = %.0f years' %(p*dt), fontsize=14)
            
            # plot elevation difference
            zfin = z[mg.nodes]
            z_diff = zfin - z_t0
            zchange = mg.node_vector_to_raster(z_diff, flip_vertically=True)
            plt.sca(ax[1])  # Set current axis
            im = plt.imshow(zchange, cmap='cividis', vmin=-0.8, vmax=0.8)
            # Add colorbar for all frames
            cbar = fig.colorbar(im, ax=ax[1], label='$\Delta$ z (m)', orientation='horizontal', fraction=0.036)
            cbar.ax.tick_params(labelsize=12)
            cbar.set_label('$\Delta$ z (m)', size=14)
            ax[1].set_xticklabels([])
            ax[1].set_yticklabels([])
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            
            # plot slope distribution
            slope_t = mg.calc_slope_at_node(z)
            slope_t = np.array(slope_t)
            info_loss = estimate_degradation_coefficient(slope_t0, slope_t, 0, ax)
            ax[2].set_xlabel('Slope', fontsize=14)
            ax[2].set_ylabel('')
            ax[2].set_yticks([])
            ax[2].set_yscale('log')
            ax[2].set_xlim([0,1])
            # Add legend for all frames
            ax[2].legend(fontsize=12)
            # Update title font size
            ax[2].set_title(r"$\phi$ = {:.2f}".format(info_loss), fontsize=14)
            
            # Add scalebar to first frame
            if p == 0:
                scalebar = ScaleBar(0.5, units="m", dimension="si-length", label=None)
                ax[0].add_artist(scalebar)
            
            # Save frame with consistent parameters
            frame_path = os.path.join(frames_dir, f'frame_{frame_counter:04d}.png')
            plt.savefig(frame_path, bbox_inches='tight', dpi=dpi, pad_inches=0)
            plt.close()
            frame_counter += 1
        
        # Update topography - using the same method as plot_evolution_time_linear
        g = mg.calc_grad_at_link(z)
        qs[mg.active_links] = -D * g[mg.active_links]
        dzdt = -mg.calc_flux_div_at_node(qs)
        z[mg.core_nodes] += dzdt[mg.core_nodes] * dt
    
    # Create GIF using imageio
    frames = []
    for i in range(frame_counter):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        frame = imageio.imread(frame_path)
        frames.append(frame)
    
    # Save GIF with 500ms duration per frame
    gif_path = f'gifs_evolution/{DEM}_evolution.gif'
    imageio.mimsave(gif_path, frames, duration=500)  # 500 milliseconds per frame
    
    # Clean up frames
    for i in range(frame_counter):
        frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
        if os.path.exists(frame_path):
            os.remove(frame_path)
    
    # Remove frames directory if it exists and is empty
    if os.path.exists(frames_dir) and not os.listdir(frames_dir):
        os.rmdir(frames_dir)
    
    return None