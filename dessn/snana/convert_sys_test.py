import os
import inspect
import numpy as np
from dessn.snana.convert_snana_dump import load_fitres, is_pos_def


def digest_folder(folder):
    filename = folder + os.sep + "FITOPT000.FITRES"
    res = load_fitres(filename)
    if res.dtype.names[0] != "CID":
        res = load_fitres(filename, skiprows=5)
    finals = []
    for row in res:
        cid = row['CID']
        z = row['zHD']
        mb = row['mB']
        x0 = row['x0']
        x1 = row['x1']
        c = row['c']

        mbe = row["mBERR"]
        x1e = row["x1ERR"]
        ce = row["cERR"]

        sim_mb = row["SIM_mB"]
        sim_x1 = row["SIM_x1"]
        sim_c = row["SIM_c"]

        cov_x1_c = row["COV_x1_c"]
        cov_x0_c = row["COV_c_x0"]
        cov_x1_x0 = row["COV_x1_x0"]

        cmbx1 = -5 * cov_x1_x0 / (2 * x0 * np.log(10))
        cmbc = -5 * cov_x0_c / (2 * x0 * np.log(10))

        cov = np.array([[mbe * mbe, cmbx1, cmbc], [cmbx1, x1e * x1e, cov_x1_c], [cmbc, cov_x1_c, ce * ce]])
        if np.any(np.isnan(cov)) or not is_pos_def(cov):
            continue
        existing_prob = 0
        finals.append([cid, z, existing_prob, sim_mb, sim_x1, sim_c, mb, x1, c] + cov.flatten().tolist())
    finals = np.array(finals)
    return finals


def digest_sys(base_folder):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    dump_dir = os.path.abspath(dir_name + "/../framework/simulations/snana_sys_dump/" + base_folder)
    output_dir_passed = os.path.abspath(dir_name + "/../framework/simulations/sys_data/" + base_folder)

    if not os.path.exists(dump_dir):
        print("Path does not exist: %s" % dump_dir)
        exit(1)

    if not os.path.exists(output_dir_passed):
        os.makedirs(output_dir_passed)

    folders = os.listdir(dump_dir)
    for f in folders:
        result = digest_folder(dump_dir + os.sep + f)
        sys_index = 0 if "STATONLY" in f else int(f.split("SYST")[1].split("_")[0])
        index = int(f.split("-")[-1])
        new_filename = output_dir_passed + os.sep + "%d_%d.npy" % (sys_index, index)
        np.save(new_filename, result.astype(np.float32))
        print(new_filename)

if __name__ == "__main__":
    digest_sys("des")
    digest_sys("lowz")
