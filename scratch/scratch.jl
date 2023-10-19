using ArraysOfArrays
using ElasticArrays
using BenchmarkTools


testvec = [sizehint!(ElasticArray{Float64, 1}(undef, 0), 1000) for i in 1:1000]

testvec2 = [sizehint!(Vector{Float64}(), 1000) for i in 1:1000]


function app!(v, n)
    for i in 1:length(v)
        for j in 1:n
            append!(v[i], 0.0)
        end
    end
end

@btime app!(testvec, 1000)
@btime app!(testvec2, 1000) # vector of vectors is slightly faster