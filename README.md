# Physics-informed nonlinear vector autoregressive models for the prediction of dynamical systems
## J. H. Adler, S. Hocking, X. Hu, S. Islam

This repository contains source code and related instructions for the purpose of reproducing the figures and results contained in the article mentioned above. While these files will remain as-is, any substantial developments will be posted to a more general repository.

# Pre-requisites
- Julia language (https://julialang.org/downloads/)

# Contents
- data : Julia scripts to generate the reference data for each test problem, and the resulting files
- modules : Julia modules for test problem parameters (Odes.jl) and piNVAR implementation (Pinvar.jl)
- scenarios : bash scripts to run scenarios for main results tables
- scripts : Julia scripts to install required packages, main results scenarios, and figures