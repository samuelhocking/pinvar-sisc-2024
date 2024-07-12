using Pkg

packages = [
    "LinearAlgebra",
    "PyPlot",
    "DataFrames",
    "DelimitedFiles",
    "Statistics",
    "Combinatorics",
    "Dates",
    "Random"
]

for p in packages
    Pkg.add(p)
end