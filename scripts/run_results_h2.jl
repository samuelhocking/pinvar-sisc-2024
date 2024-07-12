# julia run_results.jl [ode_name] [modules_dir] [data_dir] [write_dir]

using LinearAlgebra
using PyPlot
using DataFrames
using DelimitedFiles
using Statistics
using Combinatorics
using Dates

ode_name = ARGS[1]
modules_path = ARGS[2]
push!(LOAD_PATH, modules_path)
pushfirst!(DEPOT_PATH, modules_path)
data_path = ARGS[3]
write_path = ARGS[4]

using Pinvar
using Odes

# Import parameters

ode_params      = @eval $(Symbol("$(ode_name)_params"))
d               = ode_params.d
dt              = ode_params.dt
ode_rhs         = ode_params.rhs
T               = 100000

imp             = readdlm("$(data_path)/data_$(ode_name)_T=$(T)_dt=$(dt).csv",',',Float64)

println("data loaded...")

data            = imp[:,2:end]
target          = imp[2:end,2:end]

train_start     = 2001
train_end       = 3500
train_indices   = range(train_start, train_end)
rec_t_fwd = 10000
rec_starts = [
    10001,
    20001,
    30001,
    40001,
    50001
]

train_length = length(train_indices)
num_starts = length(rec_starts)

maxes = vec(1.1 * maximum(abs.(data),dims=1)')

# nonsmooth support functions
ad_factor = 1.0
bc_factor = 0.95

nonsmooth_supp_func_arr = [
    piecewise_linear_bump(-m,-bc_factor*m,bc_factor*m,m) for m in maxes
]
nonsmooth_d_supp_func_arr = [
    d_piecewise_linear_bump(-m,-bc_factor*m,bc_factor*m,m) for m in maxes
]

n = 2
k = 10
s = 1
l = Int(k*d)

monomial_basis_nonsmooth = FunctionBasis(
    monomial_funcs(nonsmooth_supp_func_arr, k),
    d_monomial_funcs(nonsmooth_d_supp_func_arr, k)
)

x_powers = [zero(I); matrix_poly_multiindices(k*d,1,n)]
phi_powers = [I; matrix_poly_multiindices(k*d,1,n)]

nsmh = monomial_h(monomial_basis_nonsmooth, x_powers, phi_powers)
nsmgh = monomial_grad_h(monomial_basis_nonsmooth, x_powers, phi_powers)

w_d             = 1.0
s_arr           = [1]
reg_arr         = [1e-12, 1e-8, 1e-4, 1e-2, 1e-1]
w_o_arr         = [0.0, 1e-4, 1e-2, 1e-1, 0.5, 1.0]
# reg_arr         = [1e-12]
# w_o_arr         = [0.0, 1e-4]

num_s = length(s_arr)
num_reg = length(reg_arr)
num_w_o = length(w_o_arr)

total_scens     = length(s_arr)*length(reg_arr)*length(w_o_arr)

data_choice     = data
target_choice   = target

vt_threshold    = 1e-4

h = nsmh
grad_h = nsmgh
h_name = "h2"

s_vec           = zeros(eltype(s_arr), total_scens)
reg_vec         = zeros(eltype(reg_arr), total_scens)
w_o_vec         = zeros(eltype(w_o_arr), total_scens)
vt_mat          = zeros(Float64, total_scens, num_starts)
E_mat           = zeros(Float64, total_scens, num_starts)
med_vt_vec      = zeros(Float64, total_scens)
med_E_vec       = zeros(Float64, total_scens)

counter = 1

println("starting grid loops...")
s = s_arr[1]
for i=1:num_reg
    reg = reg_arr[i]
    model = makeNVARModel(d, k, s, reg, dt, h, grad_h, ODE_func=ode_rhs)
    for j = 1:num_w_o
        w_o = w_o_arr[j]
        model_E_arr = zeros(num_starts)
        W = train(model, data_choice, target_choice, train_indices; data_weight=w_d, ODE_weight=w_o, verbose=false)
        model.w = W
        rec_outs = [
            recursivePredict(model, data_choice[start-model.s*(model.k-1):start,:], rec_t_fwd, return_f_pred=true) for start in rec_starts
        ]
        for p in range(1,num_starts)
            vt_mat[counter, p] = validTime(vt_threshold)(rec_outs[p][1], data[rec_starts[p]+1:rec_starts[p]+rec_t_fwd,:])
            E_mat[counter, p] = E_T(rec_outs[p][1], rec_outs[p][2], ode_rhs, dt)
        end
        s_vec[counter] = s
        reg_vec[counter] = reg
        w_o_vec[counter] = w_o
        med_vt_vec[counter] = median(vt_mat[counter,:])
        med_E_vec[counter] = median(E_mat[counter,:])
        println("done $(counter)/$(total_scens) s=$(s) reg=$(reg) w_o=$(w_o) | median vt=$(med_vt_vec[counter]) median E=$(med_E_vec[counter])")
        global counter += 1
    end
end

results = [s_vec reg_vec w_o_vec vt_mat E_mat med_vt_vec med_E_vec]

nowtime = now()

writedlm("$(write_path)/grid_results_$(ode_name)_h=$(h_name).csv",results,',')
