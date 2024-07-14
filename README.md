# Physics-informed nonlinear vector autoregressive models for the prediction of dynamical systems
### J. H. Adler, S. Hocking, X. Hu, S. Islam

This repository contains source code and related instructions for the purpose of reproducing the figures and results contained in the article mentioned above. While these files will remain as-is, any substantial developments will be posted to a more general repository.

#### Prerequisites
- Julia language (https://julialang.org/downloads/)

#### Contents
- data : Julia scripts to generate the reference data for each test problem, and the resulting files
- figures : Julia scripts to generate the article's figures, and the resulting images
- modules : Julia modules for test problem parameters (Odes.jl) and piNVAR implementation (Pinvar.jl)
- scenarios : bash scripts to run scenarios for main results tables
- scripts : Julia scripts to install required packages and run main results scenarios

#### Instructions

Clone this repo into the location of your choice.
```
git clone https://github.com/samuelhocking/pinvar-sisc-2024.git
```
Run `packages.jl` to install required Julia packages. From the `pinvar-sisc-2024/scripts` folder enter the following in a terminal window:
```
julia packages.jl
```
There are six main results tables in the article, Tables 2-7. Nine scripts generate these results: three state functions (h1, h2, and h3) for the three test problems. Each script generates a .csv file of results. The second-to-last column gives the median valid time, and the last column gives the median discrete energy. For completeness, the column schema is:
```
[
    s (skip)
    reg (regularization)
    w_o (ODE training weight)
    valid time trial 1
    valid time trial 2
    valid time trial 3
    valid time trial 4
    valid time trial 5
    discrete energy trial 1
    discrete energy trial 2
    discrete energy trial 3
    discrete energy trial 4
    discrete energy trial 5
    median valid time
    median discrete energy
]
```
To produce all of the results, navigate to the `scenarios` folder and run the following commands:
```
bash run_results_spring_h1.sh
bash run_results_spring_h2.sh
bash run_results_spring_h3.sh
bash run_results_lv_h1.sh
bash run_results_lv_h2.sh
bash run_results_lv_h3.sh
bash run_results_lorenz_h1.sh
bash run_results_lorenz_h2.sh
bash run_results_lorenz_h3.sh
```
It is advised that each of these commands is run concurrently in separate terminal windows. The resulting .csv files will be saved to the `scenarios/results` subdirectory. To produce the article's images, navigate to the `figures` folder and run:
```
julia make_figure_1.jl
julia make_figure_2.jl
```
The resulting images will be saved within the `figures` directory.