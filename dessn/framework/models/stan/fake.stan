data {

    // Declaring array and data sizes
    int<lower=0> n_sne; // Number of supernovae
    int<lower=0> n_z; // Number of redshift points
    int<lower=0> n_simps; // Number of points in simpsons algorithm

    int<lower=0>  n_surveys; // How many surveys we are analysing
    int<lower=0>  survey_map [n_sne]; // A bit from supernova to survey
    int<lower=0>  n_calib; // How many calibration

    // The input summary statistics from light curve fitting
    vector[3] obs_mBx1c [n_sne]; // SALT2 fits
    matrix[3,3] obs_mBx1c_cov [n_sne]; // Covariance of SALT2 fits
    real shift_deltas [n_sne]; // Amount of c to shift dependent on smearing model

    // Input redshift data, assumed perfect redshift for spectroscopic sample
    real <lower=0> redshifts[n_sne]; // The redshift for each SN.

    // Input ancillary data
    real <lower=0.0, upper = 1.0> prob_ia [n_sne]; // Prob of type ia
    real <lower=-1.0, upper = 1.0> masses [n_sne]; // Normalised mass estimate
    real <lower=1.0, upper = 1000.0> redshift_pre_comp [n_sne]; // Precomputed function of redshift for speed

    // Helper data used for Simpsons rule.
    real <lower=0> zs[n_z]; // List of redshifts to manually integrate over.
    real <lower=0> zsom[n_z]; // Precomputed (1+zs)^3
    real <lower=0> zspo[n_z]; // Precomputed (1+zs)
    int redshift_indexes[n_sne]; // Index of supernova redshifts (mapping zs -> redshifts)

    // Data for redshift nodes
    int num_nodes; // Num redshift nodes
    vector[num_nodes] node_weights [n_sne]; // Each supernova's node weight

    // Approximate correction in mB
    real mB_mean_orig [n_surveys];
    real mB_width_orig [n_surveys];
    real mB_alpha_orig [n_surveys];
    real mB_sgn_alpha [n_surveys];
    real mB_norm_orig [n_surveys];
    matrix[4, 4] mB_cov [n_surveys];
    int correction_skewnorm [n_surveys];
    real frac_shift;

    // Calibration std
    matrix[3, n_calib] deta_dcalib [n_sne]; // Sensitivity of summary stats to change in calib

    real <lower = 0, upper = 3> outlier_MB_delta;
    matrix[3, 3] outlier_dispersion;

    real systematics_scale; // Use this to dynamically turn systematics on or off
    int apply_efficiency;
    int apply_prior;
    int lock_systematics;

    real fakes [nsne];
}


parameters {
    real w;
}
transformed parameters {
    real posterior;
    posterior = normal_lpdf(fakes | w, 1);
}
model {
    target += posterior;
}