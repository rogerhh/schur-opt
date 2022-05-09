#!/bin/bash  
# CHOOSE TARGET

exec_roger_run="./build/driver.exe"
exec_simon_run="./build/driver_simon.exe"

machine="a10"

case $machine in
  "a10")
    NUM_MAX_THREADS=32
    ;;

  "jetson")
    NUM_MAX_THREADS=8
    ;;
esac


# profile scaling omp threads across all dataset
cd ..

# build binary
(cd build && make)

# change num omp threads
for (( num_thread=1; num_thread<=$NUM_MAX_THREADS; num_thread++ ))
do  
#    echo "Run experiments with $num_thread omp threads"
    echo "Roger: individual blocking"
    $exec_roger_run -n $num_thread
    echo "Simon: dense GEMM"
    $exec_simon_run -n $num_thread
done
