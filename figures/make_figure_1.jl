using LinearAlgebra
using PyPlot
using DataFrames
using CSV
using DelimitedFiles
using Statistics
using Random

modules_dir = "../modules"
data_path = "../data"
push!(LOAD_PATH, modules_dir)
pushfirst!(DEPOT_PATH, modules_dir)

using Pinvar
using Odes

ode_name = "lorenz"

# Make Figure 1

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
rec_start = 40001

maxes = vec(1.1 * maximum(abs.(data),dims=1)')

# smooth support functions
smooth_supp_func_arr = [
    tanhBump(m,5) for m in maxes
]
smooth_d_supp_func_arr = [
    dtanhBump(m,5) for m in maxes
]

n = 2
k = 10
s = 1
l = Int(k*d)

monomial_basis_smooth = FunctionBasis(
    monomial_funcs(smooth_supp_func_arr, k),
    d_monomial_funcs(smooth_d_supp_func_arr, k)
)

x_powers = [zero(I); matrix_poly_multiindices(k*d,1,n)]
phi_powers = [I; matrix_poly_multiindices(k*d,1,n)]

smh = monomial_h(monomial_basis_smooth, x_powers, phi_powers)
smgh = monomial_grad_h(monomial_basis_smooth, x_powers, phi_powers)

w_d = 1.0
k = 10
s = 1
reg = 1e-2
w_o = 1e-2

data_choice     = data
target_choice   = target

vt_threshold    = 1e-2

h = smh
grad_h = smgh

model = makeNVARModel(d, k, s, reg, dt, h, grad_h, ODE_func=ode_rhs)
W = train(model, data_choice, target_choice, train_indices; data_weight=w_d, ODE_weight=w_o, verbose=false)
model.w = W
rec_out = recursivePredict(model, data_choice[rec_start-model.s*(model.k-1):rec_start,:], rec_t_fwd)

vt = validTime(vt_threshold)(rec_out, data[rec_start+1:rec_start+rec_t_fwd,:])
println("vt=$(vt)")

show_t_fwd = rec_t_fwd

fig, ax = plt.subplots(d, figsize=(8,3.5), sharex=true)
for i in range(1,d)
    ax[i].plot(rec_out[1:show_t_fwd,i], label=L"model")
    ax[i].plot(data[rec_start:rec_start+show_t_fwd,i], label=L"target")
    ax[i].axvline(vt,color="grey",linestyle="dashed",label=L"t_{valid}\ M=%$(vt_threshold)")
end
ax[1].set_title(L"\text{Recursive prediction }x_1")
ax[2].set_title(L"x_2")
ax[3].set_title(L"x_3")
plt.xlabel(L"j")
fig.tight_layout()
plt.savefig("../figures/figure_1.png", bbox_inches="tight")