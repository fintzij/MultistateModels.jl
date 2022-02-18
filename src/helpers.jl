##########################
### TODO: (2022-Feb-15) Revisit when we do censoring/measurement error
##########################
"""
    collate_parameters(hazards::Vector{_Hazard})

Collate model parameters into a single vector.
"""

function collate_parameters(hazards::Vector{_Hazard})
    
    vcat([hazards[i].parameters for i in eachindex(hazards)]...)

end

"""
    set_parameter_views(parameters::Vector{Float64},hazards::Vector{_Hazard})

Set parameters in hazards objects to be views of a vector that collates all model parameters. This way we only need to modify the collated vector to update parameters.
"""

function set_parameter_views!(parameters::Vector{Float64}, hazards::Vector{_Hazard})

    start = 1

    for h in eachindex(hazards)

        npars = length(hazards[h].parameters)

        hazards[h].parameters = 
            view(parameters, start:(start + npars - 1))

        start += npars
    end
end


### for example
# foo = [2]

# # this does not modify
# function dub(a::Vector{Integer})
#     a *= 2
# end
# dub(foo)
# foo 

# # this does
# function double(a::AbstractArray{<:Number})
#     for i = firstindex(a):lastindex(a)
#         a[i] *= 2
#     end
# end
# double(foo)
# foo

# is this a problem
# function double2(a::Array)
#     for i in eachindex(a)
#         a[i] *= 2
#     end
# end

# b = collect(1:10)
# double(b) # this works
# b
# double(b[1:3]) # this doesn't
# b