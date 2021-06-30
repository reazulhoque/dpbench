#!/usr/bin/env python

# Copyright 2021 NVIDIA Corporation
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function

import argparse
from benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=100,
        dest="I",
        help="number of iterations to run",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=100,
        dest="N",
        help="number of elements in one dimension",
    )
    parser.add_argument(
        "-t",
        "--time",
        dest="timing",
        action="store_true",
        help="perform timing",
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        type=int,
        default=1,
        dest="benchmark",
        help="number of times to benchmark this application (default 1 "
        "- normal execution)",
    )
    parser.add_argument(
        "-u",
        "--use",
        default='numpy',
        choices=['numpy', 'dask', 'ramba', 'torch', 'heat', 'nums',],
        dest="use",
        help="use given numpy implementation",
    )
    args = parser.parse_args()

    if args.use == 'numpy':
        from jstencil.jstencil_numpy import run_jstencil
    elif args.use == 'dask':
        from jstencil.jstencil_dask import run_jstencil
    elif args.use == 'ramba':
        from jstencil.jstencil_ramba import run_jstencil
    elif args.use == 'torch':
        from jstencil.jstencil_torch import run_jstencil
    elif args.use == 'heat':
        from jstencil.jstencil_heat import run_jstencil
    elif args.use == 'nums':
        from jstencil.jstencil_nums import run_jstencil

    run_benchmark(
        run_jstencil,
        args.benchmark,
        f"STENCIL,{args.use}",
        (args.N, args.I, args.timing)
    )
