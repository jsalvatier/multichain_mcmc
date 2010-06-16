import timeit 


t = timeit.Timer ("from numpy import single, float_, longfloat; float_dtypes = [float, single, float_, longfloat]; single in float_dtypes and float_ in float_dtypes")
print t.timeit(50000)
from numpy import *

ones("Sd")

