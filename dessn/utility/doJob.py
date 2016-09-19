import os


def write_jobscript(filename, queue="low.q", num_cpu=24, num_walks=24):

    directory = os.path.dirname(os.path.abspath(filename))
    executable = os.path.basename(filename)
    name = executable[:-3]
    output_dir = directory + os.sep + "out_files"
    error_dir = output_dir + os.sep + "errors"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    template = '''#!/bin/bash
#$ -S /bin/bash
#$ -pe threaded 1
#$ -M samuelreay@gmail.com
#$ -N %s
#$ -m abe
#$ -q %s
#$ -V
#$ -t 1:%d
#$ -tc %d
#$ -wd %s
#$ -o %s/$JOB_NAME.$JOB_ID.out
#$ -e %s/errors
IDIR=%s
export PATH=$HOME/miniconda/bin:$PATH
source activate sam35

export OMP_NUM_THREADS="1" # set this for OpenMP threads control
export MKL_NUM_THREADS="1" # set this for Intel MKL threads control
echo 'running with OMP_NUM_THREADS =' $OMP_NUM_THREADS
echo 'running with MKL_NUM_THREADS =' $MKL_NUM_THREADS
echo 'running with NSLOTS=' $NSLOTS # number of SGE calcs
PROG=%s
PARAMS=`expr $SGE_TASK_ID - 1`
cd $IDIR
python $PROG $PARAMS'''

    n = "%s/jobscript_%s.q" % (directory, executable[:executable.index(".py")])
    t = template % (name, queue, num_walks, num_cpu + 1, output_dir, output_dir, output_dir, directory, executable)
    with open(n, 'w') as f:
        f.write(t)
    print(n)
    return n
