import numpy as np
import kokkos.gpu.core as kokkos

"Test of Kokkos Wrapper"

n = 1000000
a = np.arange(1, n+1, dtype='float64')
kokkos.start(0)
result = kokkos.reduction(a)
print(result)
kokkos.end()



