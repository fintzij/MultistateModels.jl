using MultistateModels
using Test

# setup file for generating model objects 
include("setup.jl")

#@testset "MultistateModels.jl" begin
#    # Write your tests here.
#    @test true
#end

expstruct = MultistateModels._Exponential(:exptest, [:rate], [1, 1], [1])
@test isa(MultistateModels.call_haz(t=1.0, rowind=1, expstruct; give_log = true), Float64)
