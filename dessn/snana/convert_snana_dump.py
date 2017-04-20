# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:42:49 2016

@author: shint1
"""
import numpy as np
import pandas as pd
import os
from scipy.stats import norm
import inspect

from numpy.lib.recfunctions import drop_fields


def load_fitres(filename):
    dataframe = pd.read_csv(filename, sep='\s+', skiprows=6, comment="#")
    data = dataframe.to_records()
    data = drop_fields(data, "index")
    data = drop_fields(data, "VARNAMES:")
    data = drop_fields(data, "IDSURVEY")
    data = drop_fields(data, "TYPE")
    data = drop_fields(data, "FIELD")
    data = drop_fields(data, "CUTFLAG_SNANA")
    data = drop_fields(data, "zCMBERR")
    data = drop_fields(data, "zHDERR")
    data = drop_fields(data, "zCMB")
    data = drop_fields(data, "VPEC")
    data = drop_fields(data, "VPEC_ERR")
    data = drop_fields(data, "HOST_LOGMASS")
    data = drop_fields(data, "HOST_LOGMASS_ERR")
    data = drop_fields(data, "SNRMAX1")
    data = drop_fields(data, "SNRMAX2")
    data = drop_fields(data, "SNRMAX3")
    data = drop_fields(data, "PKMJD")
    data = drop_fields(data, "PKMJDERR")
    data = drop_fields(data, "NDOF")
    data = drop_fields(data, "FITCHI2")
    data = drop_fields(data, "FITPROB")
    data = drop_fields(data, "SIM_TYPE_INDEX")
    data = drop_fields(data, "SIM_NONIA_INDEX")
    data = drop_fields(data, "SIM_NGEN_LIBID")
    data = drop_fields(data, "SIM_LIBID")
    data = drop_fields(data, "SIM_ZCMB")
    data = drop_fields(data, "SIM_DLMAG")
    data = drop_fields(data, "SIM_PKMJD")
    data = drop_fields(data, "SIM_alpha")
    data = drop_fields(data, "SIM_beta")
    data = drop_fields(data, "SIM_x0")
    data = drop_fields(data, "SIM_AV")
    return data


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def convert(base_folder):
    file = os.path.abspath(inspect.stack()[0][1])
    dir_name = os.path.dirname(file)
    dump_dir = os.path.abspath(dir_name + "/../framework/simulations/snana_dump/" + base_folder)
    output_dir_passed = os.path.abspath(dir_name + "/../framework/simulations/snana_data/" + base_folder)

    print(dump_dir)
    print(os.listdir(dump_dir))
    for folder in os.listdir(dump_dir):
        print("Digesting folder ", folder)
        folder_num = folder
        folder = os.path.abspath(dump_dir + "/" + folder)

        inner_files = list(os.listdir(folder))
        dump_file = [i for i in inner_files if i.endswith(".DUMP")][0]
        dump_file = folder + "/" + dump_file

        print("Reading %s" % dump_file)
        dataframe = pd.read_csv(dump_file, sep='\s+', skiprows=1, comment="#")
        supernovae = dataframe.to_records()

        supernovae = drop_fields(supernovae, "S2x0")
        supernovae = drop_fields(supernovae, "S2alpha")
        supernovae = drop_fields(supernovae, "S2beta")
        supernovae = drop_fields(supernovae, "VARNAMES:")
        supernovae = drop_fields(supernovae, "index")

        supernovae_passed = np.zeros(supernovae["Z"].shape)
        supernovae_cids = supernovae["CID"]
        print("Dump has fields ", supernovae.dtype)

        fitres_files = sorted([folder + "/" + i for i in inner_files if i.endswith(".FITRES")])
        base_fitres = fitres_files[0]
        mag_offset = fitres_files[1:int(len(fitres_files) // 2) + 1]
        wavelength_offset = fitres_files[int(len(fitres_files) // 2) + 1:]

        base_fits = load_fitres(base_fitres)
        mags = [load_fitres(m) for m in mag_offset]
        waves = [load_fitres(m) for m in wavelength_offset]
        mag_sort_indexes = [np.argsort(m['CID']) for m in mags]
        waves_sort_indexes = [np.argsort(m['CID']) for m in waves]
        magcidss = [m['CID'][s] for m, s in zip(mags, mag_sort_indexes)]
        wavecids = [m['CID'][s] for m, s in zip(waves, waves_sort_indexes)]

        num_bad_calib = 0
        num_bad_calib_index = np.zeros(len(mags) + len(waves))
        final_results = []
        passed_indexes = []
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

            # offsets = np.zeros((3, 8))

            offset_mb = []
            offset_x1 = []
            offset_c = []
            for mag, sorted_indexes, magcids in zip(mags + waves, mag_sort_indexes + waves_sort_indexes, magcidss + wavecids):

                index = np.searchsorted(magcids, cid)
                if index >= magcids.size or magcids[index] != cid:
                    offset_mb.append(np.nan)
                    offset_x1.append(np.nan)
                    offset_c.append(np.nan)
                else:
                    offset_mb.append(mag['mB'][sorted_indexes[index]] - mb)
                    offset_x1.append(mag['x1'][sorted_indexes[index]] - x1)
                    offset_c.append(mag['c'][sorted_indexes[index]] - c)

            if np.any(np.isnan(offset_mb)):
                num_bad_calib += 1
                num_bad_calib_index += np.isnan(offset_mb)
                continue
            offsets = np.vstack((offset_mb, offset_x1, offset_c)).T

            # Get log probabilitiy
            index = np.where(cid == supernovae['CID'])[0][-1]
            mag_smear = supernovae["MAGSMEAR_COH"][index]
            simmb = supernovae["S2mb"][index]
            simx1 = supernovae["S2x1"][index]
            simc = supernovae["S2c"][index]

            existing_prob = norm.logpdf(mag_smear, 0, 0.1) + norm.logpdf(simx1, 0, 1) + norm.logpdf(simc, 0, 0.1)

            final_result = [cid, z, existing_prob, simmb + mag_smear, simx1, simc, mb, x1, c] \
                           + cov.flatten().tolist() + offsets.flatten().tolist()
            final_results.append(final_result)
            passed_indexes.append(cid)

        supernovae_passed = np.zeros(supernovae["Z"].size, dtype=bool)
        supernovae_passed[passed_indexes] = True
        print("%d supernova passed out of %d" % (supernovae_passed.sum(), supernovae_passed.size))

        fitted_data = np.array(final_results).astype(np.float32)
        print("Truncation from %d dump -> %d dump passed -> %d fitres -> %d calibration (%d failed calib)" %
              (supernovae["Z"].shape[0], np.unique(supernovae["CID"]).size, base_fits.shape[0], fitted_data.shape[0], num_bad_calib))
        print(num_bad_calib_index)
        print(supernovae["Z"].shape, supernovae_passed.shape)
        all_mbs = np.vstack((supernovae["Z"], supernovae["S2mb"] + supernovae["MAGSMEAR_COH"], supernovae_passed)).T
        np.save(output_dir_passed + "/passed_%s.npy" % folder_num, fitted_data)
        np.save(output_dir_passed + "/all_%s.npy" % folder_num, all_mbs.astype(np.float32))

if __name__ == "__main__":
    convert("gauss0p2")
    convert("gauss0p4")
    convert("skewed0p3")
    convert("gauss0p3")
