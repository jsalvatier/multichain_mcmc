import pymc
from numpy import*

y = ones((5,10))

x = array(5)

bj = pymc.CommonDeterministics.broadcast_jacobian

print bj(y, x, y).shape