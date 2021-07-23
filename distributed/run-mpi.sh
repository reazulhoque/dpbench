#!/bin/bash
set -x

APP="$*"

echo "$NUM_NODES nodes from `hostname`"
worker_num=$NUM_NODES
hostfile=$(pwd)/mpi.hf
nodes=$(scontrol show hostnames $NODE_LIST > $hostfile)

for i in $(seq 1 $worker_num); do
    echo "$RUNNING on $i nodes"
    cmd="mpirun -n $i -f $hostfile -ppn 1 $APP --no-nodes=$i"
    echo "$cmd"
    time $cmd
    echo "return code: $?"
done

echo Done
