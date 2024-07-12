module Odes

export spring_solution
export spring_rhs
export lv_rhs
export lorenz_rhs

export Params

export spring_args
export lv_args

export spring_params
export lv_params
export lorenz_params

function spring_solution(k::Float64)
    function eval(t::Float64)
        return [
            sin(sqrt(k)*t),
            sqrt(k)*cos(sqrt(k)*t)
        ]
    end
    return eval
end

function spring_rhs(k::Float64)
    function eval(x::AbstractArray)
        return [
            x[2],
            -k*x[1]
        ]
    end
    return eval
end

function lv_rhs(a::Float64,b::Float64,c::Float64,d::Float64)
    function eval(x::AbstractArray)
        return [
            a*x[1]-b*x[1]*x[2],
            c*x[1]*x[2]-d*x[2]
        ]
    end
    return eval
end

function lorenz_rhs(x::Vector)
    a = 10
    b = 28
    c = 8/3
    return [
        a*(x[2]-x[1]),
        x[1]*(b-x[3])-x[2],
        x[1]*x[2]-c*x[3]
    ]
end

struct Params
    d::Int
    int_dt::Float64
    dt::Float64
    rhs::Function
end

spring_args = [3.0]
lv_args = [0.25,1.0,0.5,0.125]

spring_params = Params(
    2,
    1e-3,
    1e-3,
    spring_rhs(spring_args...)
)
lv_params = Params(
    2,
    1e-4,
    1e-1,
    lv_rhs(lv_args...)
)
lorenz_params = Params(
    3,
    1e-5,
    1e-3,
    lorenz_rhs
)

end