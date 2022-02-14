using Chain
using MultistateModels
using Test

# setup file for generating model objects 
# include("setup.jl") # this could be included in the individual .jl files

include("test/setup_3state_expwei.jl")
include("test/setup_3state_weiph.jl")

@testset "runtests" begin
    include("test/test_modelgeneration.jl")
    include("test/test_hazards.jl")
end

