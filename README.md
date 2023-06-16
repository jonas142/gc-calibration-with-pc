# GC calibration with PC


In this file we will explain the installation process to get the code running.
For a detailed description of the functionality and thoughts behind this code, please refer to the thesis.
## Getting started
- `pip install pybullet`
- `pip install matplotlib`
- `pip install tqdm`
- `pip install numpy`

## Setup and from_data

You can find the data from the thesis in the folder  `data/data_orientation_(1|2|t_1|t_2)`. To load that data simply copy the contents in to the respective folder found in `data/` (e.g. gc_model). Make sure to set the variable **from_data=True**. 

When running experiments, double check where the data will be saved and avoid that it overwrites your previous runs.


## Running a simulation

There is a couple of parameters you can alter. 
- `continuous_fdbk` switches feedback loop on or off
- `use_noisy_xy_speed` adds noise to the self-motion info, used to test calibration mechanism
- `env_model` T-shape or large-box
- `t-episode` for the thesis we set this to 8000
- `training` is found in `system/bio_model/parameters_fdbkLoop.py`

### Training
We turn off noisy speed to train the feedback loop (continuous_fdbk=True) in a 8000 t-episode run. And set training to True

### Calibration
Then we turn the noise on and training to False to evaluate the calibration performance. You should not switch env_model between training and testing. The results will be shown in the folder `data/plots/`. 

### plotting
for place cell tests and plots, checkout the git branch place_cell_testing