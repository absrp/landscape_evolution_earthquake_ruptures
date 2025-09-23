import numpy as np
from landlab.plot.imshow import imshow_grid
from landlab.io import read_esri_ascii
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import geopandas as gpd
from matplotlib_scalebar.scalebar import ScaleBar
from landlab.components import TaylorNonLinearDiffuser
from shapely.geometry import LineString, Point
from scipy.interpolate import RegularGridInterpolator
from osgeo import gdal


# Author: Alba M. Rodriguez Padilla
def plot_evolution_time_linear(
    n_iter,
    DEM,
    shapefiles_fault_traces,
    shapefiles_FZW,
    epsg_code,
    save_YN,
    initial_slopes,
    D=0.001,
):
    fig, ax = plt.subplots(len(n_iter), 4, tight_layout=False, figsize=(5, 9), dpi=300)
    # set overall title
    fig.suptitle(str(DEM))
    plt.subplots_adjust(
        left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.05, hspace=0.1
    )

    # to save in run
    coeff_t = []
    years_t = []
    line_length = []  # for later plot
    line_width = []  # for later plot

    # landlab grid from DEM
    DEM_name = "DEMS/" + DEM + ".asc"
    mg, z = read_esri_ascii(DEM_name, name="topographic__elevation")
    mg.set_closed_boundaries_at_grid_edges(True, True, True, True)
    slope_t0 = mg.calc_slope_at_node(z)
    slope_t0 = np.array(slope_t0)
    z_t0 = z[mg.nodes]

    # model set-up for 2D linear diffusion
    dt = 0.2 * mg.dx * mg.dx / D  # default time step is 50 years
    qs = mg.add_zeros("sediment_flux", at="link")

    plot_counter = 0
    for p in range(max(n_iter) + 1):
        if np.any(p == n_iter):
            if p * dt < dt:
                ValueError("The total time is smaller than the time-step!!")
            #print(f"iteration {p}, time = {p * dt:.1f} years")  # for debugging
            # compute hillshade
            hillshade = mg.calc_hillshade_at_node(elevs=z, alt=30.0, az=100.0)
            hillshade_raster = mg.node_vector_to_raster(hillshade, flip_vertically=False)

            # compute slope raster
            slope_t = mg.calc_slope_at_node(z)
            slope_t = np.array(slope_t)
            slope_raster = mg.node_vector_to_raster(slope_t, flip_vertically=False)

            # 68th percentile threshold
            percentile_68 = np.percentile(slope_t[~np.isnan(slope_t)], 68)
            mask = slope_raster >= percentile_68

            # Compute slopes above 68th percentile
            mask = slope_raster >= percentile_68

            # Map these slopes to 68–100 percentile scale
            slope_percentile = np.zeros_like(slope_raster)
            slope_percentile[mask] = np.interp(
                slope_raster[mask],                     # values to convert
                (percentile_68, slope_raster[mask].max()),  # from min–max slope above 68th
                (68, 100)                               # to percentiles
            )

            # Use slope_percentile to get colormap
            cmap = cm.plasma
            norm = plt.Normalize(vmin=68, vmax=100)
            slope_colored = cmap(norm(slope_percentile))
            slope_colored[..., -1] = (slope_percentile - 68) / (100 - 68) * 0.8  # alpha 0–0.6


            # plot in column 1
            ax[plot_counter, 0].imshow(
                hillshade_raster,
                cmap='gray',
                origin='lower',
                extent=extent_real,
                zorder=1
            )

            ax[plot_counter, 0].imshow(
                slope_colored,
                origin='lower',
                extent=extent_real,
                zorder=2
            )
            ax[plot_counter, 0].set_xticks([])
            ax[plot_counter, 0].set_yticks([])
            ax[plot_counter, 0].set_aspect('equal')
            ax[plot_counter, 0].set_title(f"t = {p*dt:.0f} years", fontsize=6)

            if plot_counter == 0:
                cbar = fig.colorbar(
                    cm.ScalarMappable(cmap=cmap, norm=norm),
                    ax=ax[plot_counter, 0],
                    orientation='horizontal',  # horizontal colorbar under the plot
                    fraction=0.04,            # width of the colorbar relative to the axes
                    pad=0.06                   # distance from the axes
                )
                cbar.set_label("Slope percentile", fontsize=7)
                cbar.ax.tick_params(labelsize=6)
                cbar.set_ticks([68, 84, 95, 100])  # optional tick marks

            # plot elevation difference between t and t0
            zfin = z[mg.nodes]
            z_diff = zfin - z_t0
            zchange = mg.node_vector_to_raster(z_diff, flip_vertically=True)
            im = ax[plot_counter, 2].imshow(
                zchange, cmap="cividis", vmin=-0.8, vmax=0.8
            )
            if plot_counter == 0:
                fig.colorbar(
                    im,
                    ax=ax[plot_counter, 2],
                    label="$\Delta$ z (m)",
                    orientation="horizontal",
                    fraction=0.036,
                )
            colorbar = plt.gci().colorbar
            ax[plot_counter, 2].set_yticks([])
            ax[plot_counter, 2].set_xticks([])
            slope_t = np.array(slope_t)
            slope_t0 = np.array(slope_t0)

            # calculate degradation coefficient
            deg_coeff = estimate_degradation_coefficient(
                slope_t0, slope_t, plot_counter, ax
            )
            coeff_t.append(deg_coeff)
            ax[plot_counter, 0].set_title("t = %.0f years" % (p * dt), fontsize=6)
            years_t.append(p * dt)
            ax[plot_counter, 3].set_xlabel("Slope", fontsize=6)
            ax[plot_counter, 3].set_ylabel("")
            ax[plot_counter, 3].set_yticks([])
            ax[plot_counter, 3].set_yscale("log")
            ax[plot_counter, 3].set_xlim([0, 1])
            if plot_counter == 1:
                ax[plot_counter, 3].legend(fontsize=5)

            # plot shapefile of fault traces
            fault_traces = shapefiles_fault_traces[plot_counter]
            fault_traces = fault_traces.to_crs(epsg=epsg_code)
            fault_traces.plot(ax=ax[plot_counter, 1], linewidth=0.8, color="slategrey")

            # plot width lines
            width_lines = shapefiles_FZW[plot_counter]
            width_lines = width_lines.to_crs(epsg=epsg_code)
            width_lines.plot(
                ax=ax[plot_counter, 1], linewidth=1.0, color="darkorange", alpha=0.6
            )

            ax[plot_counter, 1].set_ylabel("")
            ax[plot_counter, 1].set_yticks([])
            ax[plot_counter, 1].set_xlabel("")
            ax[plot_counter, 1].set_xticks([])
            x_min, x_max = ax[plot_counter, 0].get_xlim()
            y_min, y_max = ax[plot_counter, 0].get_ylim()
            ax[plot_counter, 1].set_xlim(x_min, x_max)
            ax[plot_counter, 1].set_ylim(y_min, y_max)

            # measure length and width of lines in shapefile at this time step
            total_length = fault_traces.geometry.length.sum() # adding all lengths of fault traces
            width = width_lines.geometry.length.sum() 

            if width == 0:
                width = 0.1 # minimum width of 0.1 for log axis in the width evolution analysis later

            ax[plot_counter, 1].set_title(
                r"$L$ = {:.2f} m, $W$ = {:.2f} m".format(total_length, width),
                fontsize=6,
            )
            line_length.append(total_length)
            line_width.append(width)

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
        rotation=None,
    )

    ax[0, 0].add_artist(scalebar)

    if save_YN == "Yes":
        DEMname = str(DEM)
        first_char = DEMname[0]
        numeric_chars = "".join(filter(str.isdigit, DEMname))
        DEMID = first_char + numeric_chars

        txtname = "Figures/" + DEMID + "_information_loss_analysis_linear.png"
        plt.savefig(txtname)

    return line_length, line_width, coeff_t, years_t, initial_slopes


def estimate_degradation_coefficient(slope_t0, slope_t, plot_counter, ax):
    cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    cleaned_slope_t = slope_t[~np.isnan(slope_t)]

    percentile_threshold = 68
    threshold_t0 = np.percentile(cleaned_slope_t0, percentile_threshold)
    threshold_t = np.percentile(cleaned_slope_t, percentile_threshold)

    deg_coeff = threshold_t0 / threshold_t

    # handle both 1D and 2D axis arrays
    if len(ax.shape) == 1:  # 1D array (1x3 grid)
        target_ax = ax[2]  # Last axis in 1x3 grid
    else:  # 2D array (4x1 grid)
        target_ax = ax[plot_counter, 3]  # Last axis in 4x1 grid

    target_ax.hist(
        cleaned_slope_t0,
        color="teal",
        histtype="step",
        alpha=0.9,
        label="Initial slopes",
    )
    target_ax.hist(
        cleaned_slope_t,
        color="darkorange",
        histtype="step",
        alpha=0.9,
        label="Slopes at time t",
    )
    target_ax.set_title(r"$\phi$ = {:.2f}".format(deg_coeff), fontsize=6)
    return deg_coeff


def load_shapefiles_for_DEMs(DEM_dir, shapefile_fault_traces_dir, shapefile_FZW_dir):
    DEMs_shapefiles = {}

    for dem_file in os.listdir(DEM_dir):
        if dem_file.endswith(".asc"):
            dem_name = os.path.splitext(dem_file)[0]

            #  shapefile directories
            fault_traces_subdir = os.path.join(shapefile_fault_traces_dir, dem_name)
            FZW_subdir = os.path.join(shapefile_FZW_dir, dem_name)

            if os.path.isdir(fault_traces_subdir) and os.path.isdir(FZW_subdir):
                #  fault trace shapefiles (FCR_*.shp)
                fault_trace_files = os.listdir(fault_traces_subdir)
                fault_shapefiles = [
                    file
                    for file in fault_trace_files
                    if file.endswith(".shp") and file.startswith("FCR_")
                ]
                fault_shapefiles_sorted = sorted(
                    fault_shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0])
                )

                fault_shapefiles_data = []
                for shp_file in fault_shapefiles_sorted:
                    shapefile_path = os.path.join(fault_traces_subdir, shp_file)
                    shapefile = gpd.read_file(shapefile_path)
                    fault_shapefiles_data.append(shapefile)

                #  width shapefiles (width_*.shp)
                width_files = os.listdir(FZW_subdir)
                width_shapefiles = [
                    file
                    for file in width_files
                    if file.endswith(".shp") and file.startswith("width_")
                ]
                width_shapefiles_sorted = sorted(
                    width_shapefiles, key=lambda x: int(x.split("_")[1].split(".")[0])
                )

                width_shapefiles_data = []
                for shp_file in width_shapefiles_sorted:
                    shapefile_path = os.path.join(FZW_subdir, shp_file)
                    shapefile = gpd.read_file(shapefile_path)
                    width_shapefiles_data.append(shapefile)

                DEMs_shapefiles[dem_name] = {
                    "fault_traces": fault_shapefiles_data,
                    "width_lines": width_shapefiles_data,
                }

    return DEMs_shapefiles


# deg coeff evolution over time function
def func_deg_coeff(x, a, c):
    return x**a * c

# length evoluton over time function
def func_line_length(x, a, b, c):
    return b - (c * x) / (x + a)

# width evoluton over time function
def func_width_evolution(x, a, b, c):
    return a + b * np.log(1 + x / c)

def normalize_length(row, length_at_time_zero):
    return row["Length (m)"] / length_at_time_zero[row["DEM ID"]]


def estimate_degradation_coefficient_noplot(
    slope_t0, slope_t
):  # little hack for instances when I don't want to plot the histogram
    cleaned_slope_t0 = slope_t0[~np.isnan(slope_t0)]
    cleaned_slope_t = slope_t[~np.isnan(slope_t)]

    percentile_threshold = 68
    threshold_t0 = np.percentile(cleaned_slope_t0, percentile_threshold)
    threshold_t = np.percentile(cleaned_slope_t, percentile_threshold)

    deg_coeff = threshold_t0 / threshold_t
    return deg_coeff

##### functions for linear-nonlinear comparison ##### based on DiMichieli Vitturi and Arrowsmith, 2013 Matlab code
# Author: Mindy Zuckerman
def load_dem(filename):
    dataset = gdal.Open(filename)
    dem = dataset.ReadAsArray()
    geotransform = dataset.GetGeoTransform()
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    x = np.arange(width) * geotransform[1] + geotransform[0]
    y = np.arange(height) * geotransform[5] + geotransform[3]
    y = y[::-1]  # flip y coordinates
    dem = np.flipud(dem)  # flip DEM vertically
    X, Y = np.meshgrid(x, y)
    return X, Y, dem


def calc_slope_histogram(dem, delta_x, delta_y, max_slope=90):
    h_x, h_y = np.gradient(dem, delta_x, delta_y)
    grad = np.sqrt(h_x**2 + h_y**2)
    slope_deg = np.degrees(np.arctan(grad)).flatten()
    bins = np.arange(0, max_slope + 1, 1)
    counts, _ = np.histogram(slope_deg, bins=bins)
    return bins[:-1], counts


def extract_profile(X, Y, Z, line_pts):
    points = np.column_stack((X.flatten(), Y.flatten()))
    values = Z.flatten()

    # points along the profile line at native spacing
    dist = np.cumsum(
        np.r_[
            0,
            np.sqrt(
                np.diff([p[0] for p in line_pts]) ** 2
                + np.diff([p[1] for p in line_pts]) ** 2
            ),
        ]
    )
    num = int(dist[-1] / (X[0, 1] - X[0, 0]))  # match grid resolution
    xs = np.linspace(line_pts[0][0], line_pts[-1][0], num=num)
    ys = np.linspace(line_pts[0][1], line_pts[-1][1], num=num)

    profile_z = []
    profile_dist = []

    for xi, yi in zip(xs, ys):
        idx = np.argmin((points[:, 0] - xi) ** 2 + (points[:, 1] - yi) ** 2)
        profile_z.append(values[idx])
        profile_dist.append(np.sqrt((xi - xs[0]) ** 2 + (yi - ys[0]) ** 2))

    return np.array(profile_dist), np.array(profile_z)
