# Copyright (c) 2017, Nicolas P. Rougier
# Copyright (c) 2021, Intel Corporation
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import datetime
import dask.array as np
from dask.distributed import Client
from os import getenv
import numpy


def run_mandelbrot(xmin, ymin, xmax, ymax, xn, yn, itermax, horizon=2.0):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    client = Client(scheduler_file=getenv("DASK_CFG"))
    nt = numpy.sum([x for x in client.nthreads().values()])
    start = datetime.datetime.now()
    Yi, Xi = np.meshgrid(range(xn), range(xn))
    xis = Xi.shape
    yis = Yi.shape
    X = np.reshape(np.linspace(xmin, xmax, xn, dtype='float64', chunks=(xn//nt,))[Xi.flatten()], xis)
    Y = np.reshape(np.linspace(ymin, ymax, yn, dtype='float64', chunks=(yn//nt,))[Yi.flatten()], yis)
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype='int64', chunks=(C.shape[0]//nt, C.shape[1]))
    Z_ = np.zeros(C.shape, dtype='complex128', chunks=(C.shape[0]//nt, C.shape[1]))
    Xi = np.reshape(Xi, (xn * yn,)).rechunk(((xn*yn)//nt,))
    Yi = np.reshape(Yi, (xn * yn,)).rechunk(((xn*yn)//nt,))
    C = np.reshape(C, (xn * yn,)).rechunk(((xn*yn)//nt,))

    Z = np.zeros(C.shape, dtype='complex128', chunks=(C.shape[0]//nt,))
    for i in range(itermax):
        if not len(Z):
            break

        # Compute for relevant points only
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # Failed convergence
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        np.logical_not(I, I)  # np.negative(I, I) not working any longer
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    result = (Z_.T, N_.T)

    total = (datetime.datetime.now()-start).total_seconds() * 1000.0
    print("Elapsed Time: " + str(total) + " ms")
    return total
