using Pkg
Pkg.activate(".")
using MultistateModels
using Test
using DataFrames
using Random
using Statistics
using LinearAlgebra
using Printf

println("Running longtest_mcem_splines.jl with SIR=:lhs")
@time include("../MultistateModelsTests/longtests/longtest_mcem_splines.jl")
