

########################################################################
########### Save cav data set from R package msm         ###############
########### Fit multistate model to mixed data using msm ###############
########################################################################

# setup ####

library(msm)
library(tidyverse)

## artificially set the observation type to a mix of panel (1) and exactly observed (2) data ####
set.seed(0)
cav <- cav %>% mutate(obstype = sample(c(1,2), nrow(.), replace = TRUE))

## save the data set locally ####
write_csv(cav, "scratch/cav.csv")



# fit multistate model with R package msm ####

## set transition matrix ####
## state 4 is an absorbing state, and all other states communicate with each other and with state 4
Q_ind <- matrix( # need connections between states 1 and 3.
    c(0,1,1,1,
      1,0,1,1,
      1,1,0,1,
      0,0,0,0),
    nrow = 4, byrow = TRUE)

## initial values for the parameters ####
Q0 <- crudeinits.msm(state ~ years, PTNUM, data = cav, qmatrix = Q_ind)

## fit the model ####
m_mixed_12 <- msm(state ~ years, PTNUM, data = cav, qmatrix = Q0, obstype = obstype)
print(m_mixed_12)

## get the estimates and loglik ####
m_mixed_12$estimates # untransformed MLE
m_mixed_12$estimates.t # transformed MLE (natural scale)
m_mixed_12$minus2loglik # loglik
