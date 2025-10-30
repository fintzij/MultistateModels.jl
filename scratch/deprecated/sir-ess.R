
#
# Sampling-importance-resampling


#
# setup
ESS=function(w) 1/sum((w/sum(w))^2)
library(tidyverse)
M=1e4 # sample size
l=1/10 # rate of exponential proposal


#
# initial sample

# generate values from the proposal (exponential distribution)
x = rexp(M, l)
hist(x)
# sample weights
w=exp(log(2)+dnorm(x,log=T)-dexp(x,l,log=T)) # include log(2) terms since the target is the half normal on the positive line
hist(w)
mean(w) # weights have an expectation of 1 since we are using the *normalized* densities to compute the weights
#plot(x,w)
# effective sample size
ESS(w)

# compare the importance sample to the truth
ggplot() +
        geom_histogram( # histogram of weighted draws
                data=tibble(x,w),
                mapping=aes(x, y=..density.., weight=w), boundary=0
                ) +
        geom_line( # true density of half normal distribution
                data=tibble(x=seq(0,10,by=0.01),dens=dnorm(seq(0,10,by=0.01))*2), # multiply the normal density by 2 because we are working with the half normal distribution.
                mapping=aes(x,dens)
                ) +
        xlim(c(0, 5))


#
# take a subsample

# size of subsample
M_sub=round(M/2)
# number of replicates in subsample
w_sub=rmultinom(1,M_sub,w)[,1]
barplot(table(w_sub))
# discard elements with weight 0
x_sub=x[w_sub>0]
w_sub=w_sub[w_sub>0]
# effective sample size
ESS(w_sub)


ggplot() +
        geom_histogram(
                data=tibble(x_sub,w_sub),
                mapping=aes(x_sub, y=..density.., weight=w_sub), boundary=0
        ) +
        geom_line(
                data=tibble(x=seq(0,10,by=0.01),dens=dnorm(seq(0,10,by=0.01))*2),
                mapping=aes(x,dens)
        ) +
        xlim(c(0, 5))


#
# add fresh draws

# number of fresh draws
M_fresh=M/2
# fresh draws and their weights
x_fresh=rexp(M_fresh,l)
w_fresh=exp(log(2)+dnorm(x_fresh,log=T)-dexp(x_fresh,l,log=T))
# append to existing sample
x_total = c(x_sub, x_fresh)
w_total = c(w_sub, w_fresh)
hist(w_total)

ggplot() +
        geom_histogram(
                data=tibble(x_total,w_total),
                mapping=aes(x_total, y=..density.., weight=w_total), boundary=0
        ) +
        geom_line(
                data=tibble(x=seq(0,10,by=0.01),dens=dnorm(seq(0,10,by=0.01))*2),
                mapping=aes(x,dens)
        ) +
        xlim(c(0, 5))

