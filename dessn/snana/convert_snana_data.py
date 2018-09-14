# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:42:49 2016

@author: shint1
"""
import numpy as np
import pandas as pd
import os
import pickle
import inspect
import re
import fnmatch
import hashlib
import logging

from scipy.stats import norm
from scipy.stats import binned_statistic
from dessn.snana.systematic_names import get_systematic_mapping


def load_fitres(filename, skiprows=6):
    # logging.debug("Loading %s" % filename)
    if filename.endswith(".gz"):
        compression = "gzip"
    else:
        compression = None
    try:
        dataframe = pd.read_csv(filename, sep='\s+', compression=compression, skiprows=skiprows, comment="#")
    except ValueError:
        logging.error("Filename %s failed to load" % filename)
        return None
    columns = ['CID', 'IDSURVEY', 'zHD', 'HOST_LOGMASS', 'HOST_LOGMASS_ERR', 'x1',
               'x1ERR', 'c', 'cERR', 'mB', 'mBERR', 'x0', 'x0ERR',
               'COV_x1_c', 'COV_x1_x0', 'COV_c_x0', 'SIM_mB', 'SIM_x1', 'SIM_c',
               'biasCor_mB', 'biasCor_x1', 'biasCor_c']
    final_columns = [c for c in columns if c in dataframe.columns]
    data = dataframe[final_columns].to_records()
    return data


def is_pos_def(x):
    if not np.all(np.isfinite(x)):
        return False
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
    expression = re.compile("[^#]ITOPT.*\[(.*)\](?!.*FITOPT000).")
    with open(nml_file) as f:
        results = expression.findall(f.read())
    return results


def get_systematic_scales(nml_file, override=False):
    scaling = get_scaling()
    systematic_names = load_systematic_names(nml_file)
    sys_label_dict = get_systematic_mapping()
    systematic_labels = [sys_label_dict.get(n, n) for n in systematic_names]
    systematics_scales = []
    for name in systematic_names:
        scale = 1.0
        if not override:
            for n, s in scaling:
                if fnmatch.fnmatch(name, n):
                    scale = s
                    break
        else:
            logging.info("Override engaged. Setting scales to unity.")
        systematics_scales.append(scale)
    logging.debug("systemtatic scales are %s" % systematics_scales)
    return systematic_labels, systematics_scales


def get_directories(base_folder):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    dump_dir = os.path.abspath(dir_name + "/data_dump/" + base_folder)
    output_dir = os.path.abspath(dir_name + "/../framework/simulations/snana_data/") + "/"
    if base_folder.split("_")[-1].startswith("v"):
        base_folder = "_".join(base_folder.split("_")[:-1])
    nml_file = dump_dir + "/" + base_folder + ".nml"
    if not os.path.exists(nml_file):
        logging.error("Cannot find the NML file at %s" % nml_file)
        exit(1)
    return dump_dir, output_dir, nml_file


def get_realisations(base_folder, dump_dir):
    if base_folder.endswith("sys"):
        base_folder = base_folder[:-4]
    if base_folder.split("_")[-1].startswith("v"):
        base_folder = "_".join(base_folder.split("_")[:-1])
    print("Base folder for sims: %s" % base_folder)
    inner_files = sorted(list(os.listdir(dump_dir)))
    inner_paths = [dump_dir + "/" + f for f in inner_files]
    sim_dirs = [p for p, f in zip(inner_paths, inner_files) if os.path.isdir(p) and f.startswith("DES3YR")]
    logging.info("Found %d sim directories in %s" % (len(sim_dirs), dump_dir))
    return sim_dirs


def load_dump_file(sim_dir):
    filename = "SIMGEN.DAT.gz" if os.path.exists(sim_dir + "/SIMGEN.DAT.gz") else "SIMGEN.DAT"
    compression = "gzip" if filename.endswith("gz") else None
    names = ["SN", "CID", "S2mb", "MAGSMEAR_COH", "S2c", "S2x1", "Z"]
    keep = ["CID", "S2mb", "MAGSMEAR_COH", "S2c", "Z"]
    dtypes = [int, float, float, float, float]
    dataframe = pd.read_csv(sim_dir + "/" + filename, compression=compression, sep='\s+',
                            skiprows=1, comment="V", error_bad_lines=False, names=names)
    logging.info("Loaded dump file from %s" % (sim_dir + "/" + filename))
    data = dataframe.to_records()
    res = []
    for row in data:
        try:
            r = [d(row[k]) for k, d in zip(keep, dtypes)]
            res.append(tuple(r))
        except Exception:
            pass
    data = np.array(res, dtype=[('CID', np.int32), ('S2mb', np.float64), ('MAGSMEAR_COH', np.float64),
                                ("S2c", np.float64), ("Z", np.float64)])
    good_mask = (data["S2mb"] > 10) & (data["S2mb"] < 30)
    data = data[good_mask]
    return data


def digest_simulation(sim_dir, systematics_scales, output_dir, systematic_labels, load_dump=False, skip=6, biascor=None, zipped=True):

    max_offset_mB = 0.2
    ind = 0
    if "-0" in sim_dir:
        ind = int(sim_dir.split("-0")[-1]) - 1
    logging.info("Digesting index %d in folder %s" % (ind, sim_dir))

    bias_fitres = None
    if biascor is not None:
        logging.info("Biascor is %s" % biascor)
        short_name = os.path.basename(sim_dir).replace("DES3YR", "").replace("_DES_", "").replace("_LOWZ_", "")
        bias_loc = os.path.dirname(os.path.dirname(sim_dir)) + os.sep + biascor + os.sep + short_name
        bias_fitres_file = bias_loc + os.sep + "SALT2mu_FITOPT000_MUOPT000.FITRES"
        assert os.path.exists(bias_fitres_file)
        fres = load_fitres(bias_fitres_file, skiprows=5)
        bias_fitres = {"%s_%s" % (row["IDSURVEY"], row["CID"]): [row['biasCor_mB'], row['biasCor_x1'], row['biasCor_c']] for row in fres}

    inner_files = sorted(list(os.listdir(sim_dir)))
    ending = ".FITRES.gz" if zipped else ".FITRES"
    fitres_files = sorted([sim_dir + "/" + i for i in inner_files if i.endswith(ending)])
    base_fitres = fitres_files[0]
    sysematics_fitres = fitres_files[1:]
    logging.info("Have %d fitres files for systematics" % len(sysematics_fitres))

    base_fits = load_fitres(base_fitres, skiprows=skip)
    sysematics = [load_fitres(m, skiprows=skip) for m in sysematics_fitres]

    sysematics_sort_indexes = [None if m is None else np.argsort(m['CID']) for m in sysematics]
    sysematics_idss = [None if m is None else m['CID'][s] for m, s in zip(sysematics, sysematics_sort_indexes)]
    systematic_labels_save = [s for sc, s in zip(systematics_scales, systematic_labels) if sc != 0]
    num_bad_calib = 0
    num_bad_calib_index = np.zeros(len(systematic_labels_save))
    logging.debug("Have %d, %d, %d, %d systematics" %
                  (len(sysematics), len(sysematics_sort_indexes), len(sysematics_idss), len(systematics_scales)))

    final_results = []
    passed_cids = []
    all_offsets = []
    logging.debug("Have %d rows to process" % base_fits.shape)
    mass_mean = np.mean(base_fits["HOST_LOGMASS_ERR"])
    not_found = 0
    fake_mass = False
    if mass_mean == -9 or mass_mean == 0:
        logging.warning("Mass is fake")
        fake_mass = True
    for i, row in enumerate(base_fits):
        if i % 1000 == 0 and i > 0:
            logging.debug("Up to row %d" % i)
        cid = row['CID']
        survey_id = row['IDSURVEY']

        key = "%s_%s" % (survey_id, cid)
        if bias_fitres is None:
            not_found += 1
            bias_mB, bias_x1, bias_c = 0, 0, 0
        else:
            if key not in bias_fitres:
                not_found += 1
                continue
            else:
                bias_mB, bias_x1, bias_c = bias_fitres[key]

        z = row['zHD']

        mb = row['mB']
        x0 = row['x0']
        x1 = row['x1']
        c = row['c']

        mass = row['HOST_LOGMASS']
        mass_err = row['HOST_LOGMASS_ERR']
        if mass < 0:
            mass_prob = 0
        else:
            if mass_err < 0.01:
                mass_err = 0.01
            mass_prob = 1 - norm.cdf(10, mass, mass_err)
        if fake_mass:
            mass_prob = 0
        mbe = row["mBERR"]
        x1e = row["x1ERR"]
        ce = row["cERR"]

        if "SIM_mB" not in row.dtype.names:
            sim_mb = 0
            sim_x1 = 0
            sim_c = 0
        else:
            sim_mb = row["SIM_mB"]
            sim_x1 = row["SIM_x1"]
            sim_c = row["SIM_c"]

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
            if scale == 0:
                continue
            if mag is None:
                offset_mb.append(0.0)
                offset_x1.append(0.0)
                offset_c.append(0.0)
                continue
            index = np.searchsorted(magcids, cid)
            if index >= magcids.size or magcids[index] != cid:
                offset_mb.append(0.0)
                offset_x1.append(0.0)
                offset_c.append(0.0)
            else:
                offset_mb.append((mag['mB'][sorted_indexes[index]] - mb) * scale)
                offset_x1.append((mag['x1'][sorted_indexes[index]] - x1) * scale)
                offset_c.append((mag['c'][sorted_indexes[index]] - c) * scale)
        if len(offset_mb) == 0:
            offset_mb.append(0.0)
            offset_x1.append(0.0)
            offset_c.append(0.0)

        if np.any(np.isnan(offset_mb)) or np.any(np.abs(offset_mb) > max_offset_mB):
            num_bad_calib += 1
            num_bad_calib_index += np.isnan(offset_mb)
            continue
        offsets = np.vstack((offset_mb, offset_x1, offset_c))
        passed_cids.append(cid)

        if isinstance(cid, str):
            cid = int(hashlib.sha256(cid.encode('utf-8')).hexdigest(), 16) % 10 ** 8
        all_offsets.append(offsets)
        final_result = [cid, z, mass_prob, sim_mb, sim_x1, sim_c, mb, x1, c, bias_mB, bias_x1, bias_c] \
                       + cov.flatten().tolist() + offsets.flatten().tolist()
        final_results.append(final_result)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fitted_data = np.array(final_results).astype(np.float32)

    np.save("%s/passed_%d.npy" % (output_dir, ind), fitted_data)
    logging.info("Bias not found for %d  out of %d SN (%d found)" % (not_found, base_fits.shape[0], base_fits.shape[0] - not_found))
    logging.info("Calib faliures: %d in total. Breakdown: %s" % (num_bad_calib, num_bad_calib_index))

    # Save the labels out
    with open(output_dir + "/sys_names.pkl", 'wb') as f:
        pickle.dump(systematic_labels_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    if load_dump:

        supernovae = load_dump_file(sim_dir)
        all_mags = supernovae["S2mb"].astype(np.float64) + supernovae["MAGSMEAR_COH"].astype(np.float64)
        all_zs = supernovae["Z"].astype(np.float64)
        all_cids = supernovae["CID"].astype(np.int32)

        cids_dict = dict([(c, True) for c in passed_cids])

        supernovae_passed = np.array([c in cids_dict for c in all_cids])
        mask_nan = ~np.isnan(all_mags)

        all_data = np.vstack((all_mags[mask_nan] + 100 * supernovae_passed[mask_nan], all_zs[mask_nan])).T
        print(all_data.shape)
        if all_data.shape[0] > 7000000:
            all_data = all_data[:7000000, :]
        np.save(output_dir + "/all_%s.npy" % ind, all_data.astype(np.float32))
        logging.info("%d nans in apparents. Probably correspond to num sims." % (~mask_nan).sum())


def convert(base_folder, load_dump=False, override=False, skip=11, biascor=None, zipped=True):

    dump_dir, output_dir, nml_file = get_directories(base_folder)
    logging.info("Found nml file %s" % nml_file)
    systematic_labels, systematics_scales = get_systematic_scales(nml_file, override=override)
    sim_dirs = get_realisations(base_folder, dump_dir)
    version = ""
    if base_folder.split("_")[-1].startswith("v"):
        version = "_" + base_folder.split("_")[-1]
    for sim in sim_dirs:
        sim_name = os.path.basename(sim)
        if "-0" in sim_name:
            this_output_dir = output_dir + sim_name.split("-0")[0]
        else:
            this_output_dir = output_dir + sim_name
        if base_folder.endswith("sys"):
            this_output_dir += "sys"
        this_output_dir += version
        digest_simulation(sim, systematics_scales, this_output_dir, systematic_labels, load_dump=load_dump,
                          skip=skip, biascor=biascor, zipped=zipped)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="[%(funcName)20s()] %(message)s")
    # convert("DES3YR_LOWZ_COMBINED_TEXT_v8")
    convert("DES3YR_DES_COMBINED_TEXT_v8")
    # convert("DES3YR_DES_NOMINAL")
    # convert("DES3YR_LOWZ_NOMINAL")
    # convert("DES3YR_DES_BULK_v8", skip=6)
    # convert("DES3YR_LOWZ_BULK_v8", skip=6)
    # convert("DES3YR_DES_SAM_COHERENT_v8", skip=11)
    # convert("DES3YR_DES_SAM_MINUIT_v8", skip=11)
    # convert("DES3YR_LOWZ_SAM_COHERENT_v8", skip=11)
    # convert("DES3YR_LOWZ_SAM_MINUIT_v8", skip=11)
    # convert("DES3YR_LOWZ_BULK", skip=6)
    # convert("DES3YR_DES_SAMTEST", skip=11)
    # convert("DES3YR_LOWZ_SAMTEST", skip=11)
    # convert("DES3YR_DES_BHMEFF_v8", load_dump=True, skip=11, zipped=True)
    # convert("DES3YR_LOWZ_BHMEFF_v8", load_dump=True, skip=11, zipped=True)
    # convert("DES3YR_LOWZ_VALIDATION", skip=11)
    # convert("DES3YR_DES_VALIDATION", skip=11)



