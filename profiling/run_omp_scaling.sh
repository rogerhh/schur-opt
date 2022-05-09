#!/bin/bash  
# CHOOSE TARGET

exec_run="./build/driver.exe"
machine="a10"

case $machine in
  "a10")
    NUM_MAX_THREADS=32
    ;;

  "jetson")
    NUM_MAX_THREADS=8
    ;;
esac

file_strongscaling=${PWD}/strongscaling.txt

# profile scaling omp threads across all dataset
cd ..

# 1. make sure we have all the data needed generated
#python3 sample_files.py -a 
python3 sample_files.py -a

# build binary
cd build
make
cd ..

# clear file
echo "Strong Scaling Profiling on $machine running $exec_run executable" &> $file_strongscaling

# change num omp threads
for (( num_thread=1; num_thread<=$NUM_MAX_THREADS; num_thread++ ))
do  
#    echo "Run experiments with $num_thread omp threads"
    $exec_run -n $num_thread &>> $file_strongscaling
done
