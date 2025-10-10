using ParameterHandling
using NestedTuples

# make tuple
pars = (
    t1 = (
        bl = positive(1.0),
        trt = real(0.5)
    ),
    t2 = (
        shape = positive(1.2),
        scale = positive(2.0),
        linpred = (
            bl = real(-1.0),
            trt = real(0.3)
        )
    )
)

# schema
s = schema(pars)

# leaf setter
f = leaf_setter(pars)

# NOTES
# hazard could carry its own parameter schema
# model object should be a mutable struct
# easiest just to create a new named tuple each time parameters are set or update
# the parameters field itself is a named tuple, which is immutable. no need for it to be mutable as creating a new one is not intensive
