from roadrunner._roadrunner import NamedArray


def ctor_named_array(reconstructor, init_args, setstate_args, colnames, rownames):
    o = reconstructor(NamedArray, *init_args)
    o.__setstate__(setstate_args)
    o.colnames = colnames
    o.rownames = rownames
    o.__array_finalize__()
    return o


def pickle_named_array(_arr):
    dumps = list(_arr.__reduce__())
    reconstructor = dumps[0]
    init_args = dumps[1][1:]
    setstate_args = dumps[2]
    return ctor_named_array, (reconstructor, init_args, setstate_args, _arr.colnames, _arr.rownames)


# Collect info for classes we'd like to support that need pickle info
# These are registered with copyreg at every service instantiation
COPY_REGS = [(NamedArray, pickle_named_array)]
