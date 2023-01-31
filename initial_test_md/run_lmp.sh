#!/bin/bash
#SBATCH --account=def-belandl1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --job-name=MPI_test
#SBATCH --output=out.run
#SBATCH --time=0-00:30 # time (DD-HH:MM)

module load       StdEnv/2020  gcc/9.3.0  cuda/11.2.2
module load openmpi/4.0.3

mpirun -np 48 /home/zjm/interface/lmp_mpi  < in.run > out.run
