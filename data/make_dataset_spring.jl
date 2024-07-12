# julia make_dataset_[ode_name].jl  [optional:modules_dir]

using LinearAlgebra
using PyPlot
using DataFrames
using CSV
using DelimitedFiles
using Statistics
using Random

modules_dir = "./modules"
push!(LOAD_PATH, modules_dir)
pushfirst!(DEPOT_PATH, modules_dir)

using Odes

params = spring_params
soln = spring_solution(spring_args...)

T = 100000
tt = Vector((0:T) .* params.dt)

data = [tt stack(soln.(tt),dims=1)]

writedlm("data_spring_T=$(T)_dt=$(params.dt).csv",data,',')