# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import pickle
import inspect
import re
import fnmatch
import hashlib

from dessn.snana.systematic_names import get_systematic_mapping


def load_fitres(filename):
    with open(filename) as f:
        c = 0
        for line in f:
            if line.startswith("VARNAMES"):
                break
            c += 1
    dataframe = pd.read_csv(filename, sep='\s+', skiprows=c, comment="#")
    columns = ["CID", "zHD", "x0", "x0ERR", "x1", "x1ERR", "c", "cERR", "mB", "mBERR", "COV_x1_c",
               "COV_x1_x0", "COV_c_x0", "SIM_mB", "SIM_x1", "SIM_c"]
    cols = [c for c in columns if c in dataframe.columns]
    data = dataframe[cols].to_records()

    return data


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def get_scaling():
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    scale_file = dir_name + os.sep + "CREATE_COV.INPUT"
    results = []
    with open(scale_file) as f:
        for line in f:
            if line.startswith("ERRSCALE"):
                comps = line.split()
                results.append((comps[1], float(comps[3])))
    return results


def load_systematic_names(nml_file):
    expression = re.compile("[^#]ITOPT.*\[(.*)\]")
    with open(nml_file) as f:
        results = expression.findall(f.read())
    return results


def load_dump(dump_file):
    # File is formatted so badly I have to edit it myself.........
    print("Reading %s" % dump_file)
    results = []
    with open(dump_file) as f:
        for line in f:
            if not line.startswith("SN:"):
                continue
            comp = line.split()
            contents = [int(comp[1])]
            for element in comp[2:]:
                contents.append(float(element))
            results.append(contents)

    supernovae = np.core.records.fromrecords(results,
                                names='CID,S2mb,MAGSMEAR_COH',
                                formats='int32,float64,float64')
    return supernovae


def convert(base_folder, nml_file, dump_file=None):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    dump_dir = os.path.abspath(dir_name + "/fits/" + base_folder)
    output_dir_passed = os.path.abspath(dir_name + "/../framework/simulations/eff_data/" + base_folder)

    scaling = get_scaling()
    systematic_names = load_systematic_names(dump_dir + "/" + nml_file)
    sys_label_dict = get_systematic_mapping()
    systematic_labels = [sys_label_dict[n] for n in systematic_names]
    systematics_scales = []
    for name in systematic_names:
        scale = 1.0
        for n, s in scaling:
            if fnmatch.fnmatch(name, n):
                scale = s
                break
        systematics_scales.append(scale)

    print("systemtatic scales are ", systematics_scales)
    print(dump_dir)
    print(os.listdir(dump_dir))
    inner_files = list(os.listdir(dump_dir))

    supernovae = None
    if dump_file is not None:
        supernovae = load_dump(dump_dir + "/" + dump_file)

    fitres_files = sorted([dump_dir + "/" + i for i in inner_files if i.endswith(".FITRES")])
    print(fitres_files)
    base_fitres = fitres_files[0]
    sysematics_fitres = fitres_files[1:]

    base_fits = load_fitres(base_fitres)
    sysematics = [load_fitres(m) for m in sysematics_fitres]
    sysematics_sort_indexes = [np.argsort(m['CID']) for m in sysematics]
    sysematics_idss = [m['CID'][s] for m, s in zip(sysematics, sysematics_sort_indexes)]

    num_bad_calib = 0
    num_bad_calib_index = np.zeros(len(sysematics))
    final_results = []
    cids = []
    for i, row in enumerate(base_fits):
        if i % 1000 == 0:
            print("Up to row %d" % i)
        cid = row['CID']
        z = row['zHD']

        mb = row['mB']
        x0 = row['x0']
        x1 = row['x1']
        c = row['c']

        mbe = row["mBERR"]
        x1e = row["x1ERR"]
        ce = row["cERR"]

        sim_mb = row["SIM_mB"] if "SIM_mB" in row else 0
        sim_x1 = row["SIM_x1"] if "SIM_x1" in row else 0
        sim_c = row["SIM_c"] if "SIM_c" in row else 0

        cov_x1_c = row["COV_x1_c"]
        cov_x0_c = row["COV_c_x0"]
        cov_x1_x0 = row["COV_x1_x0"]

        cmbx1 = -5 * cov_x1_x0 / (2 * x0 * np.log(10))
        cmbc = -5 * cov_x0_c / (2 * x0 * np.log(10))

        cov = np.array([[mbe * mbe, cmbx1, cmbc], [cmbx1, x1e * x1e, cov_x1_c], [cmbc, cov_x1_c, ce * ce]])

        if not is_pos_def(cov):
            continue

        offset_mb = []
        offset_x1 = []
        offset_c = []
        for mag, sorted_indexes, magcids, scale in \
                zip(sysematics, sysematics_sort_indexes, sysematics_idss, systematics_scales):

            index = np.searchsorted(magcids, cid)
            if index >= magcids.size or magcids[index] != cid:
                offset_mb.append(np.nan)
                offset_x1.append(np.nan)
                offset_c.append(np.nan)
            else:
                offset_mb.append((mag['mB'][sorted_indexes[index]] - mb) * scale)
                offset_x1.append((mag['x1'][sorted_indexes[index]] - x1) * scale)
                offset_c.append((mag['c'][sorted_indexes[index]] - c) * scale)

        if np.any(np.isnan(offset_mb)):
            num_bad_calib += 1
            num_bad_calib_index += np.isnan(offset_mb)
            continue
        offsets = np.vstack((offset_mb, offset_x1, offset_c)).T

        existing_prob = 1
        cids.append(cid)
        if isinstance(cid, str):
            cid = int(hashlib.sha256(cid.encode('utf-8')).hexdigest(), 16) % 10**8

        final_result = [cid, z, existing_prob, sim_mb, sim_x1, sim_c, mb, x1, c] \
                       + cov.flatten().tolist() + offsets.flatten().tolist()
        final_results.append(final_result)

    fitted_data = np.array(final_results).astype(np.float32)

    cids = sorted(cids)
    cids_dict = dict([(c, True) for c in cids])
    if len(cids) < 5000:
        print(cids)

    if not os.path.exists(output_dir_passed):
        os.makedirs(output_dir_passed)
    np.save(output_dir_passed + "/passed.npy",  fitted_data)

    if supernovae is not None:
        passed = [c in cids_dict for c in supernovae["CID"]]
        mb = supernovae["S2mb"] + supernovae["MAGSMEAR_COH"]
        efficiency = np.vstack((mb, passed)).T
        print(efficiency.shape)
        np.save(output_dir_passed + "/all.npy", efficiency)

    with open(output_dir_passed + "/sys_names.pkl", 'wb') as f:
        pickle.dump(systematic_labels, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # convert("PS1_LOWZ_COMBINED_FITS",   "bulk/LOWZ_MATRIX.NML")
    # convert("DESALL_specType_SMP_real_snana_text",   "bulk/DES_MATRIX.NML")
    # convert("SHINTON_LOWZ_MATRIX_C11_SKEWC_SKEWX1",   "bulk/LOWZ_MATRIX.NML")
    # convert("SHINTON_LOWZ_MATRIX_C11_SYMC_SYMX1",     "bulk/LOWZ_MATRIX.NML")
    # convert("SHINTON_LOWZ_MATRIX_G10_SKEWC_SKEWX1",   "bulk/LOWZ_MATRIX.NML")
    # convert("SHINTON_LOWZ_MATRIX_G10_SYMC_SYMX1",     "bulk/LOWZ_MATRIX.NML")
    # convert("SHINTON_DES_MATRIX_C11_SKEWC_SKEWX1",    "bulk/DES_MATRIX.NML")
    # convert("SHINTON_DES_MATRIX_C11_SYMC_SYMX1",      "bulk/DES_MATRIX.NML")
    # convert("SHINTON_DES_MATRIX_G10_SKEWC_SKEWX1",    "bulk/DES_MATRIX.NML")
    # convert("SHINTON_DES_MATRIX_G10_SYMC_SYMX1",      "bulk/DES_MATRIX.NML")
    # convert("DES3Y_DES_EFF_AMG10", "fit.nml", dump_file="SIMGEN.DAT")
    # convert("DES3Y_DES_EFF_CD", "fit.nml", dump_file="SIMGEN.DAT")
    convert("DES3Y_LOWZ_EFF", "fit.nml", dump_file="SIMGEN.DAT", )


