#!/bin/bash

# Create input files for DFT over strained supercells
basefile="K_e0bcc.txt";
matl="K";etype="expansion_bcc";nat=1;
#e=(0 2 4 6 8 10 12 14 15 16 17 18 19 20 21 22 23 24 25); # exx strain %
#e=[2,4,6,8,10,12,14,15,16,17,18,19,20,21,22,23,24,25]; #eyy
#e=[2,4,6,8,10,11,12,13,14,15,16,17,18,19,20]; #biax
#for i in "${e[@]}"; do

mkdir ../${matl}_${etype}_runs
for e in `seq -35 76`; do

a=$(echo "1+0.05*$e" | bc -l);
b=$(echo "1+0.05*$e" | bc -l);
c=$(echo "1+0.05*$e" | bc -l);

echo $a
# relax input files
#cat > ${matl}_exx$exx.relax.in << EOF
cat > top << EOF
&control
    disk_io = 'none',
    prefix = '${matl}_expansion$e',
    calculation ='scf',
    outdir = './out',
    pseudo_dir = '/global/home/hpc4995/potential/'
    tstress = .true.
    tprnfor = .true.
 /
 &system
    ibrav=0,
    nat=$nat,
    ntyp=1,
    ecutwfc=50,
    occupations='smearing',
    smearing = 'gaussian',
    degauss = 0.01,

 /
 &electrons
    mixing_mode='plain',
    diagonalization='david',
/
 &ions
    ion_dynamics = 'bfgs'
 /
 &cell
 /
CELL_PARAMETERS
EOF
# Modify cell parameters according to strain values
sed -n '3,5p' $basefile > cell
awk -v a=$a -v b=$b -v c=$c '{print a*$1,b*$2,c*$3}' cell > newcell
# Modify atom positions according to strain values
#sed -n '7,7p' $basefile > pos
#awk -v a=$a -v b=$b -v c=$c '{print $1,a*$2,b*$3,c*$4}' pos > newpos
# combine all
cat top newcell  pseudo kp > ${matl}_${etype}${e}.relax.in
mkdir ../${matl}_${etype}_runs/${matl}_${etype}${e}
mv ${matl}_${etype}${e}.relax.in ../${matl}_${etype}_runs/${matl}_${etype}${e}/
rm top cell newcell #clean-up
done
