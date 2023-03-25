# RGC-fovea

**Source code corresponding to Italiano et al 2022 J. Neural Eng. https://doi.org/10.1088/1741-2552/ac72c2.**

Python 3.8.5, NEURON 7.7, and a Linux PC running Ubuntu 18.04 were utilized.
Dependencies can be installed using poetry, as described below. 
Questions, feedback, suggestions, etc. should be sent to m.italiano@unsw.edu.au.

#
## Summary of files/directories 
#


### <u>**RGC-fovea**</u>
**test_installation.sh**: a  script to generate and stimulate a (small) population. To be used immediately after download + install as to validate and ensure all files are working. 
**foveal_tiles.py:** generates human (para-)foveal populations based on configurable parameters specified within so. Saves the population's cells as text files in results/trial_XXX/cellText/ and also calls scripts to generate .hoc and .swc files at results/trial_XXX/cellHoc/ and results/trial_XXX/cellSwc/, respectively. Tile .hoc files are also saved in RGC-fovea/tiles/. 

**primate_cell_generator/:** files tasked with converting the Python-computed population into .swc and .hoc files for use in the NEURON environment.

**distributions/:** extrapolated data pertaining to eccentricity-dependent characteristics required for generating foveal populations. 

**tiles/:** population tiles for inspection/validation in the NEURON envrionment. Not used by stimulation scripts. 

**stim/:** NEURON files for extracellular stimulation. 

**stimulation_tiles.py:** drives NEURON to stimulate a target population (trial) based on configurable parameters specified within so. Response metrics are saved in .csv files (see trial_XXX/sim/runXXX/**stimulation.csv**). A record of the variables used for each generation is contained in the appropriate genXXXX folder as **stimulation-info.txt**. NEURON's time and current vectors are also saved in this gen folder as stim-time-vector.txt and stim-amp-vector.txt, respectively. 

**tps_waveforms.py:** classes and functions for writing, saving, and analysing NEURON simulation files. 

**threshold_search.py:** drives stimulation_tiles.py to determine the highest amplitude with a null response, as well as the amplitudes correspondent of the population threshold and first response with more than one activated cell (end of single-cell activation window). These amplitudes are mapped to retinal location in **results/threshold.csv**.

**helpers/:** various helper functions required for plot styling analysis. 

**.poetry.lock, pyproject.toml, requirements.txt**: files for dependency management. 
  
<br>

### <u>**RGC**</u>

**biophys/:** cell biophysics checks. 

**common/:** general, commonly used code. 

**mod/:** model mechanism files. 

<br>

### <u>**RGC-fovea/results**</u>

After generating a population, the **results/** directory will be created. This is organized as: 

**trial_XXX/:** a directory corresponding to a specific population. All simulation results concerning this population will be saved in sub-directories (runXXX, genXXXX).  

trial_XXX/**runXXX/:** a sub-directory of a trial. Runs can be used to test a population (a trial) across varying stimulation conditions, e.g., an amplitude sweep. Each run is populated by .csv files which help compare its various configurations (genXXXXs). Runs allow for separation of different investigations (e.g., hexapolar simulations can be separated from monopolar simulations).

trial_XXX/runXXX/**genXXXX/:** a sub-directory of a run. Each gen represent results from a specific simulation, i.e., a specific stimulation configuration. These folders are used to save and compare different stimulation parameters such as amplitude, stimulation waveform, ...

#

## Install dependencies
#

 1. Install NEURON using `pip3 install neuron`. A successful installation can be verified by running `neurondemo` or `nrngui`. Note that the paper utilized NEURON 7.7. 
 
 2. Set up virtual environment (e.g., as described [here](https://realpython.com/intro-to-pyenv/)). 

    For e.g., install pyenv and then run the following:

    `pyenv virtualenv 3.8.5 fovea-proj` 

    `pyenv local fovea-proj`

 3. Install poetry: ``pip install poetry``. If you do not wish to use poetry for installing dependencies, a requirements.txt file has also been included (created by poetry) but it has not been tested. 
 
 4. From RGC-fovea/, install (most) dependencies (e.g. NumPy, SciPy, matplotlib, ...) using `poetry install`.

 5. Install mcp using: `sudo apt-get install -y mmv`.

 6. Install library necessary for Qt: `sudo apt-get install libxcb-xinerama0`. 

<br>

After installing all the pre-requisites, the following script can be run (from RGC-fovea/) (without editing any parameters) to run through generation and stimulation of a small sample population of 162 cells (assuming no files have been altered): `bash test_installation.py` 

This serves to: 

1. Validate that all dependencies and files are configured and installed  properly. 
2. Demonstrates to the user what the scripts generate. 
	
Note that this test script may take a few minutes to run. The following sections detail how to use each script in greater detail. 

#

**Note that with poetry, Python files should be run within a poetry shell using:**

`poetry run python ./<file_name>.py`

#
## foveal_tiles: generating (para-)foveal populations 
#
 
 1. `foveal_tiles`: Specify the trial ID as per `TRIAL`. 

 2. `foveal_tiles`: Specify whether you want **jitter** incorporated into the model (to portray a more realistic population) by specifying `JITTER` to be `True` or `False`. 
 
 3. `foveal_tiles`: Specify the center position of the population along the horizontal meridian as per the `ECC` variable. Vertical position can be designated using `ECC_Y` but do note that this was not used in our studies. 

 4. `foveal_tiles`: Specify the patch size in regards to the length of each (square) side [mm] by changing `PATCH_SIZE`. For example, `PATCH_SIZE = 0.010` generates a 0.010 mm x 0.010 mm patch (equivalently, a 100 um x 100 um patch). 

 5. `foveal_tiles`: `SHORT_AX` is a True/False toggle. If True, the model cut axons short (sets max. length to 1250 um). This is useful for reducing simulation times and model complexity. If False, the axons are allowed to extend to the optic nerve disk. 

 6. Run the script from an IDE or by executing (from RGC-fovea/):

    ``poetry run python ./foveal_tiles.py``
    
 7. This will produce a number of representations for the foveal tile specified (.txt, .hoc, .swc) in RGC-fovea/results/trial_`TRIAL`/. Plots similar to **figure 2** will also be produced and saved under RGC-fovea/plots. 

<br>

Note that a population need not be generated for each stimulation configuration. Simulations will be saved within this population's trial directory. See the above summary of trial/run/gen for more detail. 

Other parameters like mean soma diameter and mRGC:RGC proportion are also configurable (in the section labelled 'Anatomical parameters'). These follow the above within the source code and are explained by accompanying comments. In this study, the only parameters requiring change were TRIAL, ECC and PATCH_SIZE. TEST_SMALL_SOMA and NO_DENDRITES were set to TRUE when assessing the influence of morphological factors but were otherwise set to FALSE. 

#
## foveal_tiles: generating characteristic plots (figure 1)
#
Simply running foveal_tiles.py with the *create_plots()* function will generate the plots associated with **figure 1** under the directory RGC-fovea/plots. A population does NOT need to be created alongside this. 

    if __name__ == "__main__":
        create_plots()  # create fig1 plots

#
## NEURON setup and tile visualisation 
#

1. **Before utilising NEURON (or if encountering a mod error)**, go to RGC/ and run:

    ``make clean && make`` 

2. To initialise the tile in NEURON's GUI, go to RGC/ and run: 

    ``nrngui ../RGC-fovea/tiles/foveal-tile-[TRIAL].hoc`` 

3. To also visualize the tile within NEURON, re-run step 2 with GLOBAL_HAS_GUI = 1 in RGC/common/global.hoc:
   
    
        GLOBAL_HAS_GUI = 1         // flag for displaying GUI

Note that steps 2/3 are not required to run simulations (in the following section) but are useful for initial tile validation (of cell .hoc files) and visual inspections. 

#
## Simulation of electrically evoked activity (using NEURON)
#

1. Ensure GLOBAL_HAS_GUI is set to 0 in RGC/common/global.hoc:
    
        GLOBAL_HAS_GUI = 0         // flag for displaying GUI

2. Specify settings such as amplitude, time parameters, waveform type, etc. in `stimulate_tile.py`. 

3. The default electrode radius is 5 um. To change this, edit the value elecRad in RGC-fovea/stim/stimHex-foveal.hoc (for hexapolar stimulation) and RGC-fovea/stim/stimTps-foveal.hoc (for monopolar stimulation). 

        elecRad = 5   // electrode radius (um)

4. Run `poetry run python stimulate_tile.py`. 

5. Simulation results will be recorded in each generation folder and a file with response metrics will be saved (**stimulation.csv**) within the run folder. This .csv file can be used to compare configurations. 

6. Binarized activation plots and response contours will also be saved in the generation's plot directory. Note that if there was little to no activation, the response contour will not be plotted as the KDE is not possible (a warning is raised during runtime if so). 

The file can also be called as follows (though this was not used regularly): ``poetry run python ./stimulate_tile [TRIAL] {OPTIONAL: [AMP] [NO. OF PULSES]}`` 

#
## Threshold and activation-window search 
#
For determining thresholds and windows of single-cell activation... 

1. Ensure GLOBAL_HAS_GUI is set to 0 in RGC/common/global.hoc:
    
        GLOBAL_HAS_GUI = 0         // flag for displaying GUI

2. Specify settings in `threshold_search.py` (incl. TRIAL, starting amplitude, waveform type, etc.).

3. Run `poetry run python threshold_search.py`.

4. Simulation results will be recorded in each generation folder and a file with response metrics will be saved (**stimulation.csv**) within the run folder. The current values at which the highest null response (0 cells activated), threshold (>= 1 cell activated), and activation with >1 cell are recorded in a master file (results/**thresholds.csv**).


