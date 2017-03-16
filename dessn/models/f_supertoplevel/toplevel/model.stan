data {

    // Declaring array and data sizes
    int<lower=0> n_sne; // Number of supernovae
    int<lower=0> n_z; // Number of redshift points
    int<lower=0> n_simps; // Number of points in simpsons algorithm

    // The input summary statistics from light curve fitting
    vector[3] obs_mBx1c [n_sne]; // SALT2 fits
    matrix[3,3] obs_mBx1c_cov [n_sne]; // Covariance of SALT2 fits

    // Input redshift data, assumed perfect redshift for spectroscopic sample
    real <lower=0> redshifts[n_sne]; // The redshift for each SN.

    // Input ancillary data
    real <lower=-1.0, upper = 1.0> masses [n_sne]; // Normalised mass estimate
    real <lower=1.0, upper = 1000.0> redshift_pre_comp [n_sne]; // Precomputed function of redshift for speed

    // Helper data used for Simpsons rule.
    real <lower=0> zs[n_z]; // List of redshifts to manually integrate over.
    real <lower=0> zsom[n_z]; // Precomputed (1+zs)^3
    real <lower=0> zspo[n_z]; // Precomputed (1+zs)
    int redshift_indexes[n_sne]; // Index of supernova redshifts (mapping zs -> redshifts)

    // Approximate correction in mB
    real mB_mean;
    real mB_width;
    real mB_alpha;

    // Calibration std
    vector[8] calib_std; // std of calibration uncertainty, so we can draw from regular normal
    matrix[3,8] deta_dcalib [n_sne]; // Sensitivity of summary stats to change in calib
}
transformed data {
    matrix[3, 3] obs_mBx1c_chol [n_sne];
    for (i in 1:n_sne) {
        obs_mBx1c_chol[i] = cholesky_decompose(obs_mBx1c_cov[i]);
    }
}

parameters {
    ///////////////// Underlying parameters
    // Cosmology
    real <lower = 0.1, upper = 0.99> Om;
    // real <lower = -2, upper = -0.4> w;
    // Supernova model
    real <lower = -0.2, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;

    // Other effects
    real <lower = -0.2, upper = 0.2> dscale; // Scale of mass correction
    real <lower = 0, upper = 1> dratio; // Controls redshift dependence of correction
    vector[8] calibration;

    ///////////////// Latent Parameters
    vector[3] deviations [n_sne];

    ///////////////// Population (Hyper) Parameters
    real <lower = -21, upper = -18> mean_MB;
    real <lower = -10, upper = 1> log_sigma_MB;

}

transformed parameters {
    // Back to real space
    real sigma_MB;

    // Our SALT2 model
    vector [3] model_MBx1c [n_sne];
    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];


    // Lets actually record the proper posterior values
    vector [n_sne] PointPosteriors;
    vector [n_sne] weights;
    real weight;
    real Posterior;

    // Other temp variables for corrections
    real mass_correction;

    sigma_MB = exp(log_sigma_MB);



    // -------------Begin numerical integration-----------------
    //real expon;
    //expon = 3 * (1 + w);
    for (i in 1:n_z) {
        Hinv[i] = 1./sqrt( Om * zsom[i] + (1. - Om)); // * pow(zspo[i], expon)) ;
    }
    cum_simps[1] = 0.;
    for (i in 2:n_simps) {
        cum_simps[i] = cum_simps[i - 1] + (Hinv[2*i - 1] + 4. * Hinv[2*i - 2] + Hinv[2*i - 3])*(zs[2*i - 1] - zs[2*i - 3])/6.;
    }
    for (i in 1:n_sne) {
        model_mu[i] = 5.*log10((1. + redshifts[i])*cum_simps[redshift_indexes[i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
    }
    // -------------End numerical integration---------------


    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {
        // Calculate mass correction
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        // Convert into apparent magnitude
        model_mBx1c[i] = obs_mBx1c[i] + obs_mBx1c_chol[i] * deviations[i];

        // Add calibration uncertainty
        // model_mBx1c[i] = model_mBx1c[i] + deta_dcalib[i] * (calib_std .* calibration);

        // Convert population into absolute magnitude
        model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha*model_mBx1c[i][2] - beta*model_mBx1c[i][3] + mass_correction * masses[i];
        model_MBx1c[i][2] = model_mBx1c[i][2];
        model_MBx1c[i][3] = model_mBx1c[i][3];

        weights[i] = skew_normal_lpdf(model_mBx1c[i][1] | mB_mean, mB_width, mB_alpha);
        // weights[i] = 0;

        // Track and update posterior
        PointPosteriors[i] = normal_lpdf(deviations[i] | 0, 1)
         + normal_lpdf(model_MBx1c[i][1] | mean_MB, sigma_MB);
    }
    weight = sum(weights);
    Posterior = sum(PointPosteriors) - weight
        + cauchy_lpdf(sigma_MB | 0, 2.5)
        + normal_lpdf(calibration | 0, 1);

}
model {
    target += Posterior;
}