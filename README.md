<!-- ABOUT THE PROJECT -->
## About The Project
A set of scripts to simulate the effect of surface processes on surface ruptures and quantify the information loss associated with landscape evolution over time. Includes options to simulate surface processes with linear and non-linear diffusion, implemented using open-access code landlab.

## Scripts contained
- [ ] scarp_erosion_simulation_linear.ipynb
    - [ ] A script that inputs a DEM and, under the user's choice of diffusion conditions, using the linear landlab diffusion implementation, generates a synthetic DEM that has experienced landscape evolution over a  user-defined timescale. 
    - [ ] Output 1: eroded DEM in ascii format

- [ ] scarp_erosion_simulation_nonlinear.ipynb
    - [ ] A script that inputs a DEM and, under the user's choice of diffusion conditions, using the Taylor non-linear landlab diffusion implementation for transport-limited landscape evolution, generates a synthetic DEM that has experienced landscape evolution over a  user-defined timescale. 
    - [ ] Output 1: eroded DEM in ascii format
    - [ ] Output 2: elevation difference between the linear and non-linear solutions for a given time period

- [ ] information_loss_bulk.ipynb
    - [ ] A script that inputs a set of DEMs and shapefiles and, under the user's choice of diffusion conditions, generates a synthetic DEM that has experienced landscape evolution over a user-defined timescale, and estimates the information loss at different time-steps. 
    - [ ] Output 1: hillshades showing landscape evolution, simulated by diffusion
    - [ ] Output 2: evolution of measured line length over time (meters)
    - [ ] Output 3: evolution of measured fault zone width over time (meters)
    - [ ] Output 3: difference in DEM elevations between time t and time 0 (coseismic)
    - [ ] Output 4: distribution of slopes in the landscape and their evolution over time, and computation of the degradation coefficient
Includes option to save the outputs as pdf files for each DEM in a csv file compiling all measured line lengths and degradation coefficients. Must select yes in the first cell. This notebook must be ran first to produce the output csv files required for the other notebooks to run.

- [ ] degradation_coefficient_length_evolution_comparison.ipynb
    - [ ] Output 1: plot showing the evolution of degradation coefficient, line length, and fault zone width, for each model over time, each fit by different non-linear relationships.

- [ ] soil_flux_slope_solution_comparison.ipynb
    - [ ] Output 1: plot showing the relationship between soil flux and slope under different transport laws. Includes linear diffusion, and non-linear solutions from Di Michieli Vitturi and Arrowsmith (2013) and Ganti (2013).

- [ ] comparison_linear_vs_nonlinear_diffusion.ipynb
    - [ ] Output 1: comparison of the evolution of a set of scarps under linear and non-linear diffusion implementations.
    - [ ] Output 2: scarp profiles showing difference between linear and non-linear diffusion implementations
    - [ ] Output 3: comparison of slope distributions from linear and nøn-linear diffusion implementations

- [ ] synthetic_landscape.ipynb
  - [ ] A script to test the effect of geomorphic noise renewal in the degradation coefficient
    - [ ] Output 1: simulate diffusion of scarp in synthetic landscape based on [landlab tutorial](https://landlab.readthedocs.io/en/latest/tutorials/fault_scarp/landlab-fault-scarp.html)
    - [ ] Output 2: test scarp diffusion with and without noise renewal (simulates effect of vegetation evolution and other geomorphic features unrelated to scarp evolution)
    - [ ] Output 3: scarp cross-section with and without noise renewal and corresponding degradation coefficient measurement
        
- [ ] utils.py
    - [ ] Contains functions required to run the Jupyter Notebooks above. 

## Data requirements
The following directories and contained datasets are required to run the scripts above and can be accessed at [Zenodo repository](https://zenodo.org/records/10652348)

- [ ] DEMs
  Subset DEMs from the Ridgecrest and El Mayor-Cucapah lidar datasets (R and E codes indicate event). Required to run information_loss_bulk.ipynb and comparison_linear_vs_nonlinear_diffusion.ipynb
- [ ] DEMs_linear_non_linear
  Linear and non-linear diffused DEMs to test differences, based on De Vitturi and Arrowsmith implementation. Required to run comparison_linear_vs_nonlinear_diffusion.ipynb
- [ ] Shps_fault_traces
  Mapped shapefiles for the fault traces of each diffused DEM. Required to run information_loss_bulk.ipynb
- [ ] Shps_FZW
   Mapped shapefiles for the fault zone widths of each diffused DEM. Required to run information_loss_bulk.ipynb

The following data outputs are generated in the information_loss_bulk.ipynb notebook and required to run the additional notebooks: 
- [ ] information_loss_analysis_outputs.csv. Required to run degradation_coefficient_length_evolution_comparison.ipynb
- [ ] initial_slopes.txt. Required to run soil_flux_slope_solution_comparison.ipynb
<!-- CONTACT -->
## Contact

Please report suggestions and issues:

Email: alba@caltech.edu, amrodriguezpadilla@gmail.com

Project Link: [https://github.com/absrp/landscape_evolution_earthquake_ruptures](https://github.com/absrp/landscape_evolution_earthquake_ruptures)

Manuscript Link: Coming soon! Stay tuned.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTORS -->
## Contributors

Alba M. Rodriguez Padilla (USU, Caltech)

Mindy Zuckerman (ASU)

Ramon Arrowsmith (ASU)
