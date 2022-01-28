module MultistateModels

using Chain
using DataFrames
using Distributions
using LinearAlgebra
using Quadrature
using StatsModels
using StatsFuns

# Write your package code here.
export
    @formula,
    Hazard,
    multistatemodel

# typedefs
include("common.jl")

# hazard related functions
include("hazards.jl")

# model generation
include("modelgeneration.jl")

end
