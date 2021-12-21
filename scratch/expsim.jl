# gillespie's direct algorithm to simulate - See Wilkinson book in multistate_semimarkov github repo in the references folder

# want to generalize to something like this, but start simpler with a more concrete example before
# function signature hazard(t, parameters, data)
# total_hazard(t, parameters, data)

# h12, h13, h23 both exponential, no covariates yet
# h12(t) = λ_12
# h13(t) = λ_13
# h23(t) = λ_23
# h1(t) = λ_12 + λ_13
# h2(t) = λ_23
# h3(t) = 0
# time array, state array

λ_12 = 1
λ_13 = 1.5
λ_23 = 2
λ_1 = λ_12 + λ_13
λ_2 = λ_23
λ_3 = 0

tmax = 10

t = [0.0]
s = [1]
ll = 0.0

# see Wilkinson book (p.139-140)
Qmat = [-(λ_12 + λ_13) λ_12 λ_13; 0 -λ_23 λ_23; 0 0 0]
ttot = last.(t)

keep_going = true
while(keep_going == true)

    ut = rand(1)
    tnext = solve_for_tnext(total_hazards[state_cur], rand(1))
    calc_hazards!(Λ, tcur, tnext)
    snext = sample(aweights(tmat[last(s), :]))

    ttot = ttot + tnext

    if (rate == 0) || (ttot > tmax)
        keep_going = false
    else 
        # see manuscript/multistatesemimarkov.pdf (Section 4.1)
        ll +=
            log(tmat[last(s), snext]) + tmat[last(s), snext] * tnext + 
            -sum(tmat[last(s),:]) * tnext

        push!(t, tnext)
        push!(s, snext)

    end
end

add2 = function(x,y)
    x+y
end


simone = function(hazards, total_hazards, tmat::Array{Int64}, state0::Int64, time0::Float64, tmax::Float64, parameters, data) 
    
    state_cur = state0
    time_cur = time0

    state_vec = [state_cur]
    time_vec = [time_cur]

    keep_going = true

    haz_prop = zeros(Float64, length(total_hazards)) 

    while keep_going == true
        
        # sample the next event time
        time_next = -log.(rand(1)) / total_hazards[state_cur]

        # if time_next <= tmax, save the sime and sample the next state
        if(time_next <= tmax) 
            
            # calculate the hazards
            for c ∈ axes(tmat, 2)

                # fill out the hazards
                if(tmat[state_cur, c] == 0)
                    haz_prop = 0.0
                else
                    haz_prop = hazards[tmat[state_cur, c]](time_next, parameters, data)
                end
            end

            state_next = sample(aweights(haz_prop))

        else
            keep_going = false
        end
    end

    return ()
end