#!/bin/bash
matl="K";etype="shear2_bcc";
##
for e in `seq -35 -16`; do
cat > runscript << EOF
#!/bin/bash
#SBATCH --account=def-belandl1
#SBATCH --ntasks=1
#SBATCH --time=0-2:59 # time (DD-HH:MM)
#SBATCH --mem-per-cpu=9G

module load       StdEnv/2020 inter/2020.1.217 openmpi/4.0.3
module load    quantumespresso/6.6

cd /global/scratch/hpc4995/MTP/Na/${matl}_${etype}_runs/${matl}_${etype}${e}

mpirun -np 10 pw.x < ${matl}_${etype}${e}.relax.in > ../../output/${matl}_${etype}${e}.relax.out

EOF
##
cp runscript ../${matl}_${etype}_runs/${matl}_${etype}${e}/job_${matl}_${etype}${e}.qsub
sbatch ../${matl}_${etype}_runs/${matl}_${etype}${e}/job_${matl}_${etype}${e}.qsub
done
rm runscript #clean-up