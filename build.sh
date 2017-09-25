#!/bin/bash
home_path=`eval echo ~${USERNAME}`
nvcc --cubin -arch sm_61  --default-stream per-thread mat_mult.cu

#nvcc -arch sm_61  --default-stream per-thread mat_mult.cu -o mat_mult


#nvcc --cubin -arch sm_61  --default-stream per-thread demo.cu -I.

#nvcc -arch sm_61 --default-stream per-thread demo.cu -I. -o demo

#nvcc -arch sm_61 --default-stream per-thread MatMult.cu -I. -o MatMult

export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH




