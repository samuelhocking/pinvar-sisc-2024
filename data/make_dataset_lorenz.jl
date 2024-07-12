# julia make_dataset_[ode_name].jl

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

# compatibility wrapper for time-independent ODE
function wrapper(f)
    function eval(t, x)
        return f(x)
    end
    return eval
end

function RK4(f, x0, n, dt)
    d = length(x0)
    x_array = zeros(Float64, n+1, d)
    t_array = Vector((0:n) .* dt)
    x_array[1,:] = x0
    for i=2:n+1
        x = x_array[i-1,:]
        t = t_array[i-1]
        k1 = f(t, x)
        k2 = f(t+dt/2, x+dt/2*k1)
        k3 = f(t+dt/2, x+dt/2*k2)
        k4 = f(t+dt, x+dt*k3)
        x_array[i,:] = x+dt/6*(k1+2*k2+2*k3+k4)
    end
    return [t_array x_array]
end

params = lorenz_params
x0 = [-3.0,-3.0,28.0]

T = 100000
sampling_freq = Int(round(params.dt/params.int_dt))
int_T = sampling_freq*T

tt = Vector((0:int_T) .* params.int_dt)
tt_sampled = Vector((0:T) .* params.dt)

data = RK4(wrapper(params.rhs), x0, int_T, params.int_dt)
sampled_data = data[1:sampling_freq:end,:]

writedlm("data_lorenz_T=$(T)_dt=$(params.dt).csv",sampled_data,',')