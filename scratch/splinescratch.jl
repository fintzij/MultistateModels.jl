using StatsModels
using Plots
using DataFrames
using RCall
using LinearAlgebra
using ArraysOfArrays
using ElasticArrays

# simulate data
x = collect(range(0.0, length=10001, stop=2.0*pi));
x2 = collect(0.0:0.001:2.0*pi)
y = exp.(sin.(x)+randn(length(x))) 

# using Rcall - oh shit.
using RCall
@rimport splines2 as rsplines2
@rimport stats as rstats

# two ways to do this - could import splines2
@benchmark begin
s = rsplines2.mSpline(x, df = 6)
convert(Array{Float64}, rstats.predict(s, var"newx"=x2))
end

# or do it all in R and copy back only when needed. seems faster
@benchmark begin
    R"r = splines2::mSpline($x, df = 6)"
    convert(Array{Float64}, R"predict(msr, $x2)")
end 

rs = convert(ElasticArray, rcopy(R"t(splines2::mSpline($x, degree = 4, intercept = TRUE))"))
push!(rs, rand(5))


