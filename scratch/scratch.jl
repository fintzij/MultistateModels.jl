using ArraysOfArrays
using ElasticArrays

testarr = VectorOfArrays{ElasticArray{Float64}, 1}()

append!(testarr, ElasticArray{Float64, 1}(zeros(5)))
push!(testarr, ElasticArray{Float64}(undef, 5))

append!(testarr[1], zeros(3))

samplepaths        = Vector{ElasticArray{SamplePath}}(undef, nsubj)
fill!(samplepaths, ElasticArray{SamplePath}(undef, nsubj))