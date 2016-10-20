import os
import pickle
import platform
import shutil
import socket
import sys

from dessn.models.d_simple_stan.approx.calculate_bias import add_weight_to_chain
from dessn.models.d_simple_stan.approx.run_stan import get_analysis_data, \
    get_truths_labels_significance, init_fn


if __name__ == "__main__":
    file = os.path.abspath(__file__)
    dir_name = os.path.dirname(file)
    stan_output_dir = os.path.abspath(dir_name + "/stan_output")
    output_dir = os.path.abspath(dir_name + "../output")
    t = stan_output_dir + "/stan.pkl"

    dessn_dir = file[: file.index("dessn/model")]
    sys.path.append(dessn_dir)

    # Calculate which parameters we want to keep track of
    init_pos = get_truths_labels_significance()
    params = [key[0] for key in init_pos if key[2] is not None]
    params.append("Posterior")
    params.append("sumBias")

    num_cosmology = 10
    num_walks_per_cosmology = 20
    if len(sys.argv) == 2:
        # Assuming linux environment for single thread
        i = int(sys.argv[1])
        print("Running single walker, index %d" % i)
        import pystan
        num_cosmology = i // num_walks_per_cosmology
        data = get_analysis_data(seed=num_cosmology)
        n_sne = data["n_sne"]
        num_walk = i % num_walks_per_cosmology
        t = stan_output_dir + "/ston_%d_%d.pkl" % (num_cosmology, num_walk)
        sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
        fit = sm.sampling(data=data, iter=10000, warmup=2000, chains=1, init=init_fn)
        # Dump relevant chains to file
        print("Saving chain %d" % i)
        with open(t, 'wb') as output:
            dictionary = fit.extract(pars=params)
            dictionary = add_weight_to_chain(dictionary, n_sne)
            pickle.dump(dictionary, output)
    else:
        # Run that stan locally
        p = platform.platform()
        h = socket.gethostname()
        if "smp-cluster" in h or "edison" in h:
            # Assuming this is obelix
            from dessn.utility.doJob import write_jobscript, write_jobscript_slurm
            if len(sys.argv) == 3:
                num_walks = int(sys.argv[1])
                num_jobs = int(sys.argv[2])
            else:
                num_walks = num_cosmology * num_walks_per_cosmology
                num_jobs = 30
            if os.path.exists(stan_output_dir):
                shutil.rmtree(stan_output_dir)
            os.makedirs(stan_output_dir)

            if "smp-cluster" in h:
                filename = write_jobscript(file, name=os.path.basename(dir_name),
                                           num_walks=num_walks, num_cpu=num_jobs,
                                           outdir="log", delete=True)
                os.system("qsub %s" % filename)
                print("Submitted SGE job")
            elif "edison" in h:
                filename = write_jobscript_slurm(file, name=os.path.basename(dir_name),
                                                 num_walks=num_walks, num_cpu=num_jobs,
                                                 delete=True)
                os.system("sbatch %s" % filename)
                print("Submitted SLURM job")
        else:
            print("Running short steps")
            if not os.path.exists(stan_output_dir):
                os.makedirs(stan_output_dir)
            # Assuming its my laptop vbox
            import pystan

            num_walks_per_cosmology = 1
            num_cosmology = 2
            for i in range(num_walks_per_cosmology * num_cosmology):
                num_cosmology = i // num_walks_per_cosmology
                data = get_analysis_data(seed=num_cosmology)
                n_sne = data["n_sne"]
                num_walk = i % num_walks_per_cosmology
                t = stan_output_dir + "/ston_%d_%d.pkl" % (num_cosmology, num_walk)
                sm = pystan.StanModel(file="model.stan", model_name="Cosmology")
                fit = sm.sampling(data=data, iter=3000, warmup=1000, chains=4, init=init_fn)
                # Dump relevant chains to file
                with open(t, 'wb') as output:
                    dictionary = fit.extract(pars=params)
                    dictionary = add_weight_to_chain(dictionary, n_sne)
                    pickle.dump(dictionary, output)
