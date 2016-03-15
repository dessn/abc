import os
import sys

args = sys.argv
if len(args) > 1:
    queue = args[1]
else:
    queue = "low.q"
cores = 60
if len(args) > 2:
    cores = int(args[2])


directory = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(__file__)
out_files = directory + os.sep + "out"
error_files = directory + os.sep + "error"
if not os.path.exists(out_files):
    os.mkdir(out_files)
if not os.path.exists(error_files):
    os.mkdir(error_files)

template = '''#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi %d
#$ -M samuelreay@gmail.com
#$ -N toyModel
#$ -m abe
#$ -q %s
#$ -V
#$ -l gpu=0
#$ -wd %s
#$ -o %s/$JOB_NAME.$JOB_ID.out
#$ -e %s
IDIR=%s

export OMP_NUM_THREADS="1" # set this for OpenMP threads control
export MKL_NUM_THREADS="1" # set this for Intel MKL threads control
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo 'running with NSLOTS=' $NSLOTS # number of SGE calcs

cd $IDIR
load module openmpi-x86_64
mpirun -np $NSLOTS python toyModel.py
python $PROG $PARAMS'''


n = "jobscript_%s" % queue
t = template % (cores, queue, directory, out_files, error_files, directory)
with open(n, 'w') as f:
    f.write(t)
print(n)