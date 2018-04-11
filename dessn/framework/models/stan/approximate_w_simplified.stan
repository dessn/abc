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
}
transformed data {
    matrix[3, 3] obs_mBx1c_chol [n_sne];
    matrix[4, 4] mb_cov_chol [n_surveys];
    matrix[3, 3] outlier_population;

    outlier_population = outlier_dispersion * outlier_dispersion';

    for (i in 1:n_sne) {
        obs_mBx1c_chol[i] = cholesky_decompose(obs_mBx1c_cov[i]);
    }
    for (i in 1:n_surveys) {
        mb_cov_chol[i] = cholesky_decompose(mB_cov[i]);
    }
}

parameters {
    ///////////////// Underlying parameters
    // Cosmology
    real <lower = 0.05, upper = 0.99> Om;
    real <lower = -2, upper = -0.4> w;
    // Supernova model
    real <lower = -0.1, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;

    ///////////////// Latent Parameters
    vector[3] deviations [n_sne];

    ///////////////// Population (Hyper) Parameters
    real <lower = -20.5, upper = -18.5> mean_MB;
    real <lower = -4, upper = -0.5> log_sigma_MB;


}

transformed parameters {


    real weight;
    real posterior;
    real posteriorsum;
    vector [n_surveys] survey_posteriors;
    real sigma_MB;

    {

        // Helper variables for Simpsons rule
        real Hinv [n_z];
        real cum_simps[n_simps];
        real model_mu[n_sne];

        // Lets actually record the proper posterior values
        vector [n_sne] point_posteriors;
        vector [n_sne] weights;
        vector [n_sne] numerator_weight;

        // Other temp variables for corrections
        real expon;

        // -------------Begin numerical integration-----------------
        expon = 3 * (1 + w);
        for (i in 1:n_z) {
            Hinv[i] = 1./sqrt( Om * zsom[i] + (1. - Om) * pow(zspo[i], expon));
        }
        cum_simps[1] = 0.;
        for (i in 2:n_simps) {
            cum_simps[i] = cum_simps[i - 1] + (Hinv[2*i - 1] + 4. * Hinv[2*i - 2] + Hinv[2*i - 3])*(zs[2*i - 1] - zs[2*i - 3])/6.;
        }
        for (i in 1:n_sne) {
            model_mu[i] = 5.*log10((1. + redshifts[i])*cum_simps[redshift_indexes[i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
        }
        // -------------End numerical integration---------------

        sigma_MB = exp(log_sigma_MB);

        // Now update the posterior using each supernova sample
        for (i in 1:n_sne) {

            // Convert into apparent magnitude
            model_mBx1c[i] = obs_mBx1c[i] + obs_mBx1c_chol[i] * deviations[i];

            // Convert population into absolute magnitude
            model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha * model_mBx1c[i][2] - beta * model_mBx1c[i][3];

            // Track and update posterior
            point_posteriors[i] = normal_lpdf(deviations[i] | 0, 1) + normal_lpdf(model_MBx1c[i][1] | mean_MB, sigma_MB);

        }
        posteriorsum = sum(point_posteriors);
    }
    posterior = posteriorsum + sum(survey_posteriors) + cauchy_lpdf(sigma_MB | 0, 1);
}
model {
    target += posterior;

    if (apply_prior) {
        target += normal_lpdf(Om | 0.3, 0.01);
    }
}