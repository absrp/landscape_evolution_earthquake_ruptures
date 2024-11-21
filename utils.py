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
from PIL import Image
import matplotlib.cbook as cbook
import scipy.optimize
from scipy.optimize import curve_fit
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib as mpl
from landlab.components import TaylorNonLinearDiffuser


### TO DO:
# Test evolution of 2 standard deviations of slopes 
# DEM lighting, constant limits for hillshade

def plot_evolution_time_linear(n_iter, DEM, shapefiles_input, epsg_code, save_YN, D=0.001):
    fig, ax = plt.subplots(
    len(n_iter),4,
    tight_layout=True,
    figsize=(8,10),
    dpi=300)
    
    # set overall title
    fig.suptitle(str(DEM)) 
    
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
            ax[plot_counter,0].set_title('t = %.0f years' %(p*dt),fontsize=8)
            years_t.append(p*dt)
            ax[plot_counter,3].set_xlabel('Slope',fontsize=8)
            ax[plot_counter,3].set_ylabel('')
            ax[plot_counter,3].set_yticks([])
            ax[plot_counter,3].set_yscale('log')  
            ax[plot_counter,3].set_xlim([0,1])  
            
            # plot shapefile
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
        
        g = mg.calc_grad_at_link(z)
        qs[mg.active_links] = -D * g[mg.active_links]
        dzdt = -mg.calc_flux_div_at_node(qs)
        z[mg.core_nodes] += dzdt[mg.core_nodes] * dt  
        
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
    plt.tight_layout()    
    
    if save_YN == 'Yes':
        DEMname = str(DEM)           
        first_char = DEMname[0]
        numeric_chars = ''.join(filter(str.isdigit, DEMname))
        DEMID = first_char + numeric_chars

        txtname = 'Figures/' + DEMID + '_information_loss_analysis_linear.pdf'
        plt.savefig(txtname)

    return line_length, coeff_t, years_t       
    
def plot_evolution_time_nonlinear(n_iter, DEM, shapefiles_input, epsg_code, save_YN, D=0.001):
    fig, ax = plt.subplots(
    len(n_iter),4,
    tight_layout=True,
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
            ax[plot_counter,0].set_title('t = %.0f years' %(p*dt),fontsize=8)
            years_t.append(p*dt)
            ax[plot_counter,3].set_xlabel('Slope',fontsize=8)
            ax[plot_counter,3].set_ylabel('')
            ax[plot_counter,3].set_yticks([])
            ax[plot_counter,3].set_yscale('log')  
            ax[plot_counter,3].set_xlim([0,1])  
            
            # plot shapefile
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
    plt.tight_layout()    
    
    if save_YN == 'Yes':
        DEMname = str(DEM)           
        first_char = DEMname[0]
        numeric_chars = ''.join(filter(str.isdigit, DEMname))
        DEMID = first_char + numeric_chars

        txtname =  'Figures/' + DEMID + '_information_loss_analysis_nonlinear.pdf'
        plt.savefig(txtname)

    return line_length, coeff_t, years_t

def estimate_degradation_coefficient(slope_t0,slope_t,plot_counter,ax,nbins=20):
    cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    cleaned_slope_t = slope_t[~np.isnan(slope_t)]   

    percentile_threshold = 68
    threshold_t0 = np.percentile(cleaned_slope_t0, percentile_threshold)
    threshold_t = np.percentile(cleaned_slope_t, percentile_threshold)
    
    info_loss = threshold_t0/threshold_t
    ax[plot_counter,3].hist(cleaned_slope_t0, bins=nbins, color='teal', histtype='step', alpha=0.9)
    ax[plot_counter,3].hist(cleaned_slope_t, bins=nbins, color='darkorange',histtype='step', alpha=0.9)
    ax[plot_counter,3].set_title(r"$\phi$ = {:.2f}".format(info_loss),fontsize=8)
    
    # cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    # cleaned_slope_t = slope_t[~np.isnan(slope_t)]
    # cleaned_slope_t0 = cleaned_slope_t0[cleaned_slope_t0>0]
    # cleaned_slope_t =cleaned_slope_t[cleaned_slope_t>0]
    # counts_t0, bin_edges_t0 = np.histogram(cleaned_slope_t0, bins=nbins)
    # counts_t, bin_edges_t = np.histogram(cleaned_slope_t, bins=nbins)
    # max_bin_index_t0 = np.argmax(counts_t0)
    # max_bin_edge_t0 = (bin_edges_t0[max_bin_index_t0] + bin_edges_t0[max_bin_index_t0 + 1])/2
    # max_bin_index_t = np.argmax(counts_t)
    # max_bin_edge_t = (bin_edges_t[max_bin_index_t] + bin_edges_t[max_bin_index_t + 1])/2
    # ax[plot_counter,3].hist(cleaned_slope_t0, bins=nbins, color='teal', histtype='step', alpha=0.9)
    # ax[plot_counter,3].hist(cleaned_slope_t, bins=nbins, color='darkorange',histtype='step', alpha=0.9)
    # ax[plot_counter,3].axvline(max_bin_edge_t0,color='teal',alpha=0.5)
    # ax[plot_counter,3].axvline(max_bin_edge_t,color='darkorange',alpha=0.5)
    # info_loss= max_bin_edge_t0/max_bin_edge_t
    # ax[plot_counter,3].set_title(r"$\phi$ = {:.2f}".format(info_loss),fontsize=8)
    return info_loss

def load_shapefiles_for_DEMs(DEM_dir, shapefile_dir):
    DEMs_shapefiles = {}
    
    for dem_file in os.listdir(DEM_dir):
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