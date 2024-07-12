module Pinvar

using LinearAlgebra
using Combinatorics
using DataFrames

export NVARModel
export apply_entrywise
export tanhBump
export dtanhBump
export piecewise_linear_bump
export d_piecewise_linear_bump
export FunctionBasis
export monomial_funcs
export d_monomial_funcs
export cheb_n_ab
export d_cheb_n_ab
export cheb_functions
export d_cheb_functions
export poly_multiindices
export matrix_poly_multiindices
export monomial_h
export monomial_grad_h
export cheb_h
export cheb_grad_h
export makeNVARModel
export getStateInputData
export evalVectorOfFuncs
export makeStateVector
export makeStateVector!
export makeStateMatrix
export makeGradientMatrix
export makeDlinDt
export makeChainVector
export makeChainMatrix
export train
export train!
export predict
export predict!
export recursivePredict
export recursivePredict!
export validTime
export E_T

mutable struct NVARModel
    d::Int
    k::Int
    s::Int
    l::Int
    # m::Int
    reg::Float64
    natural_dt::Float64
    h::Function
    grad_h::Union{Function,Nothing}
    ODE_func::Union{Function,Nothing}
    w::AbstractArray
    state::AbstractArray
end

function apply_entrywise(funcs,x)
    size_f = size(funcs)
    size_x = size(x)
    if size_f != size_x
        throw("funcs and x are unequal size: $(size_f) != $(size_x)")
    else
        out = zeros(size_f)
        for i in eachindex(out)
            out[i] = funcs[i](x[i])
        end
        return out
    end
end

"""
Univariate bump function that smoothly transitions from 1 to 0
# Arguments
- `m::Float64`: maximum (really x s.t. tanhBump(x)=1/2)
- `s::Int`: sharpness parameter
"""
function tanhBump(m::Float64, s::Int, t::Float64=0.0)
    function eval(x::Float64)
        return 1/2*(1+tanh((s + 1/4) * pi * (m^2 - (x-t)^2)))
    end
    return eval
end

"""
Derivative of univariate bump function that smoothly transitions from 1 to 0
# Arguments
- `m::Float64`: maximum (really x s.t. tanhBump(x)=1/2)
- `s::Int`: sharpness parameter
"""
function dtanhBump(m::Float64, s::Int)
    function eval(x::Float64)
        return -(s + 1/4) * pi * x * (sech((s + 1/4) * pi * (m^2 - x^2)))^2
    end
    return eval
end

function piecewise_linear_bump(a,b,c,d)
    function eval(x)
        if x <= a
            return 0.0
        elseif x <= b
            return (x-a)/(b-a)
        elseif x <= c
            return 1.0
        elseif x <= d
            return 1-(x-c)/(d-c)
        else
            return 0.0
        end
    end
    return eval
end

function d_piecewise_linear_bump(a,b,c,d)
    function eval(x)
        if x <= a
            return 0.0
        elseif x <= b
            return 1/(b-a)
        elseif x <= c
            return 0.0
        elseif x <= d
            return -1/(d-c)
        else
            return 0.0
        end
    end
    return eval
end

mutable struct FunctionBasis
    funcs::Union{Function,Nothing}
    d_funcs::Union{Function,Nothing}
end

function monomial_funcs(supp_funcs=nothing, repeat_count=1)
    function eval(x)
        if supp_funcs == nothing
            return x
        else
            return [
                x apply_entrywise(repeat(supp_funcs,repeat_count),x)
            ]
        end
    end
    return eval
end

function d_monomial_funcs(d_supp_funcs=nothing, repeat_count=1)
    function eval(x)
        if d_supp_funcs == nothing
            return ones(size(x))
        else
            return [
                ones(size(x)) apply_entrywise(repeat(d_supp_funcs,repeat_count),x)
            ]
        end
    end
    return eval
end

function cheb_n_ab(n,a,b)
    function eval(x)
        if x < a
            return cos(n*pi)
        elseif x < b
            return cos(n*acos((a+b-2*x)/(a-b)))
        else
            return 1.0
        end
    end
    return eval
end

function d_cheb_n_ab(n,a,b)
    function eval(x)
        y = (a+b-2*x)/(a-b)
        if x < a
            return 0.0
        elseif x == a
            return -(2*n^2)/(b-a)*cos(n*pi)
        elseif x < b
            return (-2*n*sin(n*acos(y)))/((a-b)*sqrt(1-y^2))
        elseif x == b
            return (2*n^2)/(b-a)
        else
            return 0.0
        end
    end
    return eval
end

function cheb_functions(n,a_arr,b_arr)
    d = size(a_arr,1)
    n_mat = ones(d) .* (0:n)'
    a_mat = a_arr .* ones(n+1)'
    b_mat = b_arr .* ones(n+1)'
    func_mat = Matrix{Function}(undef,size(n_mat))
    for i in eachindex(n_mat)
        func_mat[i] = cheb_n_ab(n_mat[i],a_mat[i],b_mat[i])
    end
    function eval(x)
        x_mat = x .* ones(n+1)'
        return apply_entrywise(func_mat,x_mat)
    end
end

function d_cheb_functions(n,a_arr,b_arr)
    d = size(a_arr,1)
    n_mat = ones(d) .* (0:n)'
    a_mat = a_arr .* ones(n+1)'
    b_mat = b_arr .* ones(n+1)'
    func_mat = Matrix{Function}(undef,size(n_mat))
    for i in eachindex(n_mat)
        func_mat[i] = d_cheb_n_ab(n_mat[i],a_mat[i],b_mat[i])
    end
    function eval(x)
        x_mat = x .* ones(n+1)'
        return apply_entrywise(func_mat,x_mat)
    end
end

function poly_multiindices(d, min_n, max_n; reverse=false)
    if reverse
        return cat([collect(multiexponents(d, n)) for n=(max_n:-1:min_n)]...,dims=1)
    else
        return cat([collect(multiexponents(d, n)) for n=(min_n:max_n)]...,dims=1)
    end
end

function matrix_poly_multiindices(d, min_n, max_n)
    return stack(poly_multiindices(d, min_n, max_n),dims=1)
end

function monomial_h(function_basis::FunctionBasis, x_powers, phi_powers)
    function eval(x)
        vals = function_basis.funcs(x)
        return [
            prod([vals[:,1] .^ x_powers[i,:]; vals[:,2] .^ phi_powers[i,:]]) for i = 1:size(x_powers,1)
        ]
    end
    return eval
end

function monomial_grad_h(function_basis::FunctionBasis, x_powers, phi_powers)
    function eval(x)
        vals = function_basis.funcs(x)
        d_vals = function_basis.d_funcs(x)
        m,d = size(x_powers)
        out = zeros(m,d)
        for i = 1:m
            for j = 1:d
                if x_powers[i,j] != 0
                    temp_x_powers = deepcopy(x_powers[i,:])
                    temp_dx_powers = zero(temp_x_powers)
                    coeff = Float64(temp_x_powers[j])
                    temp_x_powers[j] -= 1
                    temp_dx_powers[j] += 1
                    out[i,j] += coeff*prod([
                        vals[:,1] .^ temp_x_powers;
                        vals[:,2] .^ phi_powers[i,:];
                        d_vals[:,1] .^ temp_dx_powers
                    ])
                end
                if phi_powers[i,j] != 0
                    temp_phi_powers = deepcopy(phi_powers[i,:])
                    temp_dphi_powers = zero(temp_phi_powers)
                    coeff = Float64(temp_phi_powers[j])
                    temp_phi_powers[j] -= 1
                    temp_dphi_powers[j] += 1
                    out[i,j] += coeff*prod([
                        vals[:,1] .^ x_powers[i,:];
                        vals[:,2] .^ temp_phi_powers;
                        d_vals[:,2] .^ temp_dphi_powers
                    ])
                end 
            end
        end
        return out
    end
    return eval
end

function cheb_h(function_basis::FunctionBasis, cheb_indices)
    function eval(x)
        vals = function_basis.funcs(x)
        m,d = size(cheb_indices)
        return [
            prod([vals[i,cheb_indices[j,i]+1] for i = 1:d]) for j = 1:m
        ]
    end
end

function cheb_grad_h(function_basis::FunctionBasis, cheb_indices)
    function eval(x)
        vals = function_basis.funcs(x)
        d_vals = function_basis.d_funcs(x)
        m,d = size(cheb_indices)
        out = zeros(m,d)
        for i = 1:m
            for j = 1:d
                out[i,j] = prod([vals[l,cheb_indices[i,l]+1] for l in (1:d)[1:d .!= j]])*d_vals[j,cheb_indices[i,j]+1]
            end
        end
        return out
    end
    return eval
end

"""
Constructor for classic NVAR model with quadratic polynomial 
# Arguments
- `d::Int`: ODE dimension
- `k::Int`: lookback
- `s::Int`: skip
- `reg::Float64`: skip
- `natural_dt::Float64`: time step
"""
function makeNVARModel(d::Int, k::Int, s::Int, reg::Float64, natural_dt::Float64, h::Function, grad_h::Union{Function,Nothing}=nothing;ODE_func::Union{Function,Nothing}=nothing)
    l = Int(k*d)
    # m = Int(1 + 3/2*l + 1/2*l^2)
    return NVARModel(
        # d, k, s, l, m, reg, natural_dt,
        d, k, s, l, reg, natural_dt,
        h, grad_h, ODE_func, zeros(d, d), zeros(d)
        )
end

function getStateInputData(model::NVARModel, data::AbstractArray)
    return vec(data[end .- model.s*Vector(0:model.k-1),:]')
end

function evalVectorOfFuncs(funcs)
    function eval(x::Union{Float64,Vector})
        out = deepcopy(x)
        for i in range(1,length(x))
            out[i] = funcs[i](x[i])
        end
        return out
    end
    return eval
end

"""
Construct state vector
# Arguments
- `model::NVARModel`: reference model object
- `input::AbstractArray`: array of row vectors corresponding to model k and s=
"""
function makeStateVector(model::NVARModel, input::AbstractArray)
    # lin = vec(input')
    # return model.h(lin)
    return model.h(input)
end

function makeStateVector!(model::NVARModel, input::AbstractArray)
    model.state = makeStateVector(model, input)
end

function makeStateMatrix(model::NVARModel, data::AbstractArray, train_indices::AbstractArray)
    k = model.k
    s = model.s
    # m = model.m
    T = length(train_indices)
    out = []
    for i in range(1,T)
        t = train_indices[i]
        t_data = data[t-(k-1)*s:t,:]
        t_input = getStateInputData(model, t_data)
        push!(out, makeStateVector(model, t_input))
    end
    return stack(out)
end

function makeGradientMatrix(model::NVARModel, input::AbstractArray)
    # lin = vec(input')
    # return model.grad_h(lin)
    return model.grad_h(input)
end

function makeDlinDt(model::NVARModel, input::AbstractArray)
    k = model.k
    d = model.d
    ODE_func = model.ODE_func
    # lin = vec(input')
    # out = reshape(lin,(d,k))
    out = reshape(input,(d,k))
    for j in range(1,k)
        out[:,j] = ODE_func(out[:,j])
    end
    return vec(out)
end

function makeChainVector(model::NVARModel, input::AbstractArray)
    return makeGradientMatrix(model, input) * makeDlinDt(model, input)
end

"""
state_matrix should be unsupported to extract unsupported linear portion
"""
function makeChainMatrix(model::NVARModel, data::AbstractArray, train_indices::AbstractArray)
    k = model.k
    s = model.s
    # m = model.m
    T = length(train_indices)
    out = []
    for i in range(1,T)
        t = train_indices[i]
        t_data = data[t-(k-1)*s:t,:]
        t_input = getStateInputData(model, t_data)
        push!(out, makeChainVector(model, t_input))
    end
    return stack(out)
end

function train(
    model::NVARModel, data::AbstractArray, target::AbstractArray, train_indices::AbstractArray;
    data_weight::Float64=1.0, ODE_weight::Float64=0.0, verbose::Bool=true
    )
    reg = model.reg
    d = model.d
    ODE_func = model.ODE_func
    data_input = data[train_indices,:]
    data_output = target[train_indices,:]
    data_target = data_output - data_input
    T = length(train_indices)
    data_state = makeStateMatrix(model, data, train_indices)
    m = size(data_state,1)
    # data_target -> T x d (each point is a row vector)
    # data_state -> m x T (each state vector is a column vector)
    # W -> d x m
    # least squares problem: 
    # data_target' = W * data_state
    # data_target = data_state' * W'
    A = sqrt(data_weight)*data_state'
    b = sqrt(data_weight)*data_target
    # append regularization block if applicable
    if reg > zero(reg)
        A = [A; sqrt(reg)*I]
        b = [b; zeros(m, d)]
    end
    # append ODE block if applicable
    if (ODE_func != nothing) && (ODE_weight > zero(ODE_weight))
        chain_state = makeStateMatrix(model, data, train_indices)
        ODE_state = makeChainMatrix(model, data, train_indices)
        ODE_input = zero(data_input)
        ODE_output = zero(data_output)
        for i in range(1,T)
            ODE_input[i,:] = ODE_func(data_input[i,:])
            ODE_output[i,:] = ODE_func(data_output[i,:])
        end
        ODE_target = ODE_output - ODE_input
        A = [A; sqrt(ODE_weight)*ODE_state']
        b = [b; sqrt(ODE_weight)*ODE_target]
    end
    W = Matrix((A \ b)')              # automatically uses QR decomp
    if verbose
        println("training obj func: $(norm(A * W' - b))")
    end
    return W
end

function train!(
    model::NVARModel, data::AbstractArray, target::AbstractArray, train_indices::AbstractArray;
    data_weight::Float64=1.0, ODE_weight::Float64=0.0, verbose::Bool=true
    )
    model.w = train(model, data, target, train_indices, data_weight=data_weight, ODE_weight=ODE_weight, verbose=verbose)
end

function predict(model::NVARModel, input::AbstractArray)
    state = makeStateVector!(model, input)
    return model.w * state
end

function predict!(model::NVARModel, input::AbstractArray)
    makeStateVector!(model, input)
    return model.w * model.state
end

function recursivePredict(model::NVARModel, data::AbstractArray, t_forward::Int; return_f_pred::Bool=false)
    k = model.k
    s = model.s
    d = model.d
    old_data = data[end-(k-1)*s:end,:]
    new_data = zeros(t_forward, d)
    old_length = size(old_data,1)
    out = [old_data; new_data]
    if return_f_pred
        f_out = zeros(t_forward,d)
    end
    for t in range(1,t_forward)
        out_idx = old_length+t-1
        t_input = getStateInputData(model, out[out_idx-(k-1)*s:out_idx,:])
        state = makeStateVector(model, t_input)
        out[out_idx+1,:] = out[out_idx,:] + (model.w * state)
        if return_f_pred
            f_out[t,:] = model.ODE_func(out[out_idx,:]) + model.w * makeChainVector(model, t_input)
        end
    end
    if return_f_pred
        return out[old_length+1:end,:], f_out
    else
        return out[old_length+1:end,:]
    end
end

function recursivePredict!(model::NVARModel, data::AbstractArray, t_forward::Int; return_f_pred::Bool=false)
    k = model.k
    s = model.s
    d = model.d
    old_data = data[end-(k-1)*s:end,:]
    new_data = zeros(t_forward, d)
    old_length = size(old_data,1)
    out = [old_data; new_data]
    if return_f_pred
        f_out = zeros(t_forward,d)
    end
    for t in range(1,t_forward)
        out_idx = old_length+t-1
        t_input = getStateInputData(model, out[out_idx-(k-1)*s:out_idx,:])
        makeStateVector!(model, t_input)
        out[out_idx+1,:] = out[out_idx,:] + (model.w * model.state)
        if return_f_pred
            f_out[t,:] = model.ODE_func(out[out_idx,:]) + model.w * makeChainVector(model, t_input)
        end
    end
    if return_f_pred
        return out[old_length+1:end,:], f_out
    else
        return out[old_length+1:end,:]
    end
end

function validTime(threshold::Float64)
    function eval(prediction::AbstractArray, target::AbstractArray)
        len = size(target,1)
        valid_idx = len
        arr = sum(abs2,prediction-target,dims=2) ./ sum(abs2,target,dims=2)
        for i in range(1,len)
            if arr[i] >= threshold
                valid_idx = i
                break
            end
        end
        return valid_idx
    end
end

function E_T(x_out, f_out, f, dt)
    T = size(x_out,1)
    # return 1/(T*dt)*norm(f_out - stack([dy_dt(x_out[k,:]) for k=1:T],dims=1))^2
    return dt/2*norm(f_out - stack([f(x_out[k,:]) for k=1:T],dims=1))^2
end

end