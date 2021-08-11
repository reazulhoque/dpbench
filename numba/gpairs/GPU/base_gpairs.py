# Copyright (C) 2017-2018 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os,json
import numpy as np
import numpy.random as rnd

try:
    import itimer as it
    now = it.itime
    get_mops = it.itime_mops_now
except:
    from timeit import default_timer
    now = default_timer
    get_mops = lambda t0, t1, n: (n / (t1 - t0),t1-t0)

dtype = np.float32

SEED = 7777777
DEFAULT_NBINS = 20
DEFAULT_RMIN, DEFAULT_RMAX = 0.1, 50
DEFAULT_RBINS = np.logspace(
    np.log10(DEFAULT_RMIN), np.log10(DEFAULT_RMAX), DEFAULT_NBINS).astype(
        dtype)
DEFAULT_RBINS_SQUARED = (DEFAULT_RBINS**2).astype(dtype)

######################################################
# GLOBAL DECLARATIONS THAT WILL BE USED IN ALL FILES #
######################################################

# make xrange available in python 3
try:
    xrange
except NameError:
    xrange = range

###############################################
def get_device_selector (is_gpu = True):
    if is_gpu is True:
        device_selector = "gpu"
    else:
        device_selector = "cpu"

    if os.environ.get('SYCL_DEVICE_FILTER') is None or os.environ.get('SYCL_DEVICE_FILTER') == "opencl":
        return "opencl:" + device_selector

    if os.environ.get('SYCL_DEVICE_FILTER') == "level_zero":
        return "level_zero:" + device_selector

    return os.environ.get('SYCL_DEVICE_FILTER')

def load_data():
    x1 = np.genfromtxt('../../../data/gpairs/x1.csv', delimiter=",")
    y1 = np.genfromtxt('../../../data/gpairs/y1.csv', delimiter=",")
    z1 = np.genfromtxt('../../../data/gpairs/z1.csv', delimiter=",")
    w1 = np.genfromtxt('../../../data/gpairs/w1.csv', delimiter=",")
    x2 = np.genfromtxt('../../../data/gpairs/x2.csv', delimiter=",")
    y2 = np.genfromtxt('../../../data/gpairs/y2.csv', delimiter=",")
    z2 = np.genfromtxt('../../../data/gpairs/z2.csv', delimiter=",")
    w2 = np.genfromtxt('../../../data/gpairs/w2.csv', delimiter=",")
    rbins_squared = np.genfromtxt('../../../data/gpairs/rbins_squared.csv', delimiter=",")

    return (x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared)

def copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2):
    device_env = ocldrv.runtime.get_gpu_device()
    d_x1 = device_env.copy_array_to_device(x1.astype(dtype))
    d_y1 = device_env.copy_array_to_device(y1.astype(dtype))
    d_z1 = device_env.copy_array_to_device(z1.astype(dtype))
    d_w1 = device_env.copy_array_to_device(w1.astype(dtype))

    d_x2 = device_env.copy_array_to_device(x2.astype(dtype))
    d_y2 = device_env.copy_array_to_device(y2.astype(dtype))
    d_z2 = device_env.copy_array_to_device(z2.astype(dtype))
    d_w2 = device_env.copy_array_to_device(w2.astype(dtype))

    d_rbins_squared = device_env.copy_array_to_device(
        DEFAULT_RBINS_SQUARED.astype(dtype))

    return (
        d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared
    )

def copy_d2h(d_result):
    device_env.copy_array_from_device(d_result)

##############################################	

def run(name, alg, sizes=3, step=2, nopt=2**10):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', required=False, default=sizes,  help="Number of steps")
    parser.add_argument('--step',  required=False, default=step,   help="Factor for each step")
    parser.add_argument('--size',  required=False, default=nopt,   help="Initial data size")
    parser.add_argument('--repeat',required=False, default=1,    help="Iterations inside measured region")
    parser.add_argument('--text',  required=False, default="",     help="Print with each result")
    parser.add_argument('--json',  required=False, default=__file__.replace('py','json'), help="output json data filename")

    args = parser.parse_args()
    sizes= int(args.steps)
    step = int(args.step)
    nopt = int(args.size)
    repeat=int(args.repeat)
 
    output = {}
    output['name']      = name
    output['sizes']     = sizes
    output['step']      = step
    output['repeat']    = repeat
    output['randseed']  = SEED
    output['metrics']   = []

    rnd.seed(SEED)

    f=open("perf_output.csv",'w')
    f2 = open("runtimes.csv",'w',1)

    x1_full, y1_full, z1_full, w1_full, x2_full, y2_full, z2_full, w2_full, rbins_squared = load_data()


    for i in xrange(sizes):
        x1 = x1_full[:nopt]
        y1 = y1_full[:nopt]
        z1 = z1_full[:nopt]
        w1 = w1_full[:nopt]
        x2 = x2_full[:nopt]
        y2 = y2_full[:nopt]
        z2 = z2_full[:nopt]
        w2 = w2_full[:nopt]

        #d_x1, d_y1, d_z1, d_w1, d_x2, d_y2, d_z2, d_w2, d_rbins_squared = copy_h2d(x1, y1, z1, w1, x2, y2, z2, w2)
        iterations = xrange(repeat)

        alg(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared) #warmup
        t0 = now()
        for _ in iterations:
            alg(x1, y1, z1, w1, x2, y2, z2, w2, rbins_squared)

        mops,time = get_mops(t0, now(), nopt)
        f.write(str(nopt) + "," + str(mops*2*repeat) + "\n")
        f2.write(str(nopt) + "," + str(time) + "\n")
        print("ERF: {:15s} | NPoints: {:10d} | ITEMSIZE: {:10d}B  | MOPS: {:15.2f} | TIME: {:10.6f}".format(name, nopt, np.dtype(dtype).itemsize, mops*2*repeat,time),flush=True)
        output['metrics'].append((nopt,mops,time))
        nopt *= step
        repeat -= step
        if repeat < 1:
            repeat = 1
    json.dump(output,open(args.json,'w'),indent=2, sort_keys=True)
    f.close()
    f2.close()
