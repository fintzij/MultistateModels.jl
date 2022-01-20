using Chain
using MultistateModels
using Test

# setup file for generating model objects 
# include("setup.jl") # this could be included in the individual .jl files

include("test/setup.jl")

@testset "runtests" begin
    include("test/test_modelgeneration.jl")
end

# this sort of stuff should go in the testhazards.jl file
expstruct = MultistateModels._Exponential(:exptest, [:rate], [1, 1], [1])
@test isa(MultistateModels.call_haz(t=1.0, rowind=1, expstruct; give_log = true), Float64)
