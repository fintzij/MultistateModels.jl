# Scott's paper
#
# hidden states                     h_t in {1, ..., K} = S
# observed variables                d_t in S (exact observation) or in {C_1, C_2, C_3} (censored observation) where {C_.} is a partition of S
#
# initial distribution              pi_0(r)=Pr(X_1=r)
# Markov transition probabilities   q_t(r,s)=Pr(X_t=s|X_{t-1}=r) for t=2,...,n     [Eq. 1]
# emission distribution             P_{s}(C) = Pr(d_t=C|h_t=s)                     [Eq. 2]


# application to msm
#
# pi_0: point mass (exactly observed) or uniform (censored)
# q_t: tpm, may be time-inhomogeneous (covariates)
# P_{s}(C) = 1{s \in C}

# emission distribution
function P(s, C_t)

    s in C_t
    # s in C_t ? 1 : 0
    
end

# ffbs
function sample_skeleton(C)

    # set up
    S = 4  # size of sample space
    n = 10 # number of time steps
    m = Array{Float64}(undef, n, S) # matrix of marginal probabilities pi_t
    m[1,:] = [1,0,0,0] # initial distribution pi_0
    p = Array{Float64}(undef, n, S, S) # joint distribution Pr(h_{t-1}=r,h_t=s|d_{1:t})

    # forward filtering
    for t in 2:n
        # compute p_t
        p_tmp = Array{Float64}(undef, S, S) # unnormalized p_t
        for r in 1:S, s in 1:S
            p_tmp[r,s] = m[t-1,r] * tmp[t,r,s] * P(s, C[t]) # marginal * transition * emission [Eq. 6]
        end
        p[t,:,:] = p_tmp ./ sum(p_tmp) # normalize p_t [Eq. 6]
        # compute m_t
        for s in 1:S
            m[t,s] = sum(p[t,:,s]) # marginalize the joint distribution p_t
        end
    end

    # backward sampling
    h = Array{Int64}(undef, n)
    ## a. draw draw h_n ~ pi_n
    h[n] = Categorical(pi[n])
    ## b.  draw h_t|h_{t+1}=s ~ p_{t,.,s}
    for t in (n-1):1
        h[t] = Categorical(p[t,:,h[t+1]]) # [Eq. 10]
    end

    return h
end