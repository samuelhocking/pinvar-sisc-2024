using PyPlot

modules_dir = "../modules"
push!(LOAD_PATH, modules_dir)
pushfirst!(DEPOT_PATH, modules_dir)

using Pinvar
using Odes

# Make Figure 2 (smooth)

xx = LinRange(-1.25,1.25,1001)
m = 1.0
t = 0.0
s_arr = [1, 2, 3, 4, 5]

fig, ax = plt.subplots(2, figsize=(3.5,3.5), sharex=true, sharey=true)
for s in s_arr
    # tb = tanhBump(m,s)
    tb = tanhBump(m,s,t)
    yy = @.tb(xx)
    ax[1].plot(xx, yy, label=L"\xi=%$(s)")
    ax[2].plot(xx, yy .* (xx.^2))
end
plt.figlegend(ncol=1, loc="right", bbox_to_anchor=(1.25,0.5))
ax[1].set_title(L"\text{Smooth }\phi(x)")
ax[2].set_title(L"\text{Smooth }\phi(x)x^2")
ax[2].set_xlabel(L"x")
ax[1].set_ylabel(L"\phi(x)")
ax[2].set_ylabel(L"\phi(x)x^2")
fig.tight_layout()
plt.savefig("../figures/figure_2_smooth.png", bbox_inches="tight")

# Make Figure 2 (nonsmooth)

xx = LinRange(-1.25,1.25,1001)
ad = 1.0
bc = 0.95

pwlb = piecewise_linear_bump(-ad,-bc,bc,ad)

fig, ax = plt.subplots(2, figsize=(3.5,3.5), sharex=true, sharey=true)
yy = pwlb.(xx)
ax[1].plot(xx, yy)
ax[2].plot(xx, yy .* (xx.^2))
ax[1].set_title(L"\text{Non-smooth }\phi(x)")
ax[2].set_title(L"\text{Non-smooth }\phi(x)x^2")
ax[2].set_xlabel(L"x")
ax[1].set_ylabel(L"\phi(x)")
ax[2].set_ylabel(L"\phi(x)x^2")
fig.tight_layout()
plt.savefig("../figures/figure_2_nonsmooth.png", bbox_inches="tight")