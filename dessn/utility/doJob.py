import os
import sys

args = sys.argv

assert len(args) >= 2, "You need to supply the filename to run"
filename = args[1]
if len(args) > 2:
    queue = args[2]
else:
    queue = "low.q"
cores = 60
if len(args) > 3:
    cores = int(args[3])


directory = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.abspath(directory + os.sep + filename)
base_name = os.path.basename(file_path).split(".")[0]
name = os.path.basename(file_path)
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
#$ -N %s
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
### module load openmpi-x86_64
mpirun -np $NSLOTS python %s $NSLOTS
'''


n = "jobscript_%s_%s" % (base_name, queue)
t = template % (cores, base_name, queue, directory, out_files, error_files, directory, file_path)
with open(n, 'w') as f:
    f.write(t)
print(n)