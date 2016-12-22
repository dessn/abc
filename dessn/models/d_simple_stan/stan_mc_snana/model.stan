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
    real <lower=0.0, upper = 1.0> mass [n_sne]; // Normalised mass estimate
    real <lower=1.0, upper = 1000.0> redshift_pre_comp [n_sne]; // Precomputed function of redshift for speed

    // Helper data used for Simpsons rule.
    real <lower=0> zs[n_z]; // List of redshifts to manually integrate over.
    real <lower=0> zsom[n_z]; // Precomputed (1+zs)^3
    real <lower=0> zspo[n_z]; // Precomputed (1+zs)
    int redshift_indexes[n_sne]; // Index of supernova redshifts (mapping zs -> redshifts)

    // All data for calculating the biases
    int<lower=0> n_sim; // Number of simulated supernova
    real sim_log_prob[n_sim]; // Probability of drawing supernova from distribution
    vector[3] sim_mBx1c [n_sim]; // SALT2 fits to simulated supernova
    real <lower=0> sim_redshifts[n_sim]; // The redshift for each simulated SN.
    real <lower=0.0, upper = 1.0> sim_mass [n_sim]; // Normalised mass estimate for simulated SN.
    real <lower=1.0, upper = 1000.0> sim_redshift_pre_comp [n_sim]; // Precomputed function of redshift for speed
    int sim_redshift_indexes[n_sim]; // Index of supernova redshifts (mapping zs -> redshifts)

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
    real <lower = 0.1, upper = 1> Om;
    // real <lower = -2, upper = -0.4> w;
    // Supernova model
    real <lower = -0.2, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;

    // Other effects
    real <lower = -0.2, upper = 0.2> dscale; // Scale of mass correction
    real <lower = 0, upper = 1> dratio; // Controls redshift dependence of correction

    ///////////////// Latent Parameters
    vector[3] deviations [n_sne];
    //real <lower = -21, upper = -18> true_mB[n_sne];
    //real <lower = -8, upper = 8> true_x1[n_sne];
    //real <lower = -1, upper = 2> true_c[n_sne];

    ///////////////// Population (Hyper) Parameters
    real <lower = -21, upper = -18> mean_MB;
    real <lower = -0.5, upper = 0.5> mean_x1;
    real <lower = -0.2, upper = 0.2> mean_c;
    real <lower = 0.001, upper = 0.3> sigma_MB;
    real <lower = 0.001, upper = 2> sigma_x1;
    real <lower = 0.001, upper = 0.4> sigma_c;
    cholesky_factor_corr[3] intrinsic_correlation;

}

transformed parameters {
    // Our SALT2 model
    vector [3] model_MBx1c [n_sne];
    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    // Our SALT2 model for simulated supernova
    vector [3] sim_MBx1c [n_sim];


    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];
    real sim_model_mu[n_sim];

    // Modelling intrinsic dispersion
    matrix [3,3] population;
    vector [3] mean_MBx1c;
    vector [3] sigmas;


    // Lets actually record the proper posterior values
    vector [n_sne] PointPosteriors;
    vector [n_sim] bias_correction;
    real weight;
    real Posterior;

    // Other temp variables for corrections
    real mass_correction;

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
    for (i in 1:n_sim) {
        sim_model_mu[i] = 5.*log10((1. + sim_redshifts[i])*cum_simps[sim_redshift_indexes[i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
    }
    // -------------End numerical integration---------------

    // Calculate SN population
    mean_MBx1c[1] = mean_MB;
    mean_MBx1c[2] = mean_x1;
    mean_MBx1c[3] = mean_c;
    sigmas[1] = sigma_MB;
    sigmas[2] = sigma_x1;
    sigmas[3] = sigma_c;
    population = diag_pre_multiply(sigmas, intrinsic_correlation);

    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {
        // Calculate mass correction
        // mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        // Convert into apparent magnitude
        model_mBx1c[i] = obs_mBx1c[i] + obs_mBx1c_chol[i] * deviations[i];

        // Convert population into absolute magnitude
        model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha*model_mBx1c[i][2] - beta*model_mBx1c[i][3];// + mass_correction * mass[i];
        model_MBx1c[i][2] = model_mBx1c[i][2];
        model_MBx1c[i][3] = model_mBx1c[i][3];

        // Track and update posterior
        PointPosteriors[i] = normal_lpdf(deviations[i] | 0, 1) + multi_normal_cholesky_lpdf(model_MBx1c[i] | mean_MBx1c, population);
    }

    // Calculate the weights
    for (i in 1:n_sim) {
        mass_correction = dscale * (1.9 * (1 - dratio) / sim_redshift_pre_comp[i] + dratio);
        sim_MBx1c[i][1] = sim_mBx1c[i][1] - sim_model_mu[i] + alpha*sim_mBx1c[i][2] - beta*sim_mBx1c[i][3];// + mass_correction * sim_mass[i];
        sim_MBx1c[i][2] = sim_mBx1c[i][2];
        sim_MBx1c[i][3] = sim_mBx1c[i][3];

        bias_correction[i] = multi_normal_cholesky_lpdf(sim_MBx1c[i] | mean_MBx1c, population) - sim_log_prob[i];
    }
    weight = n_sne * log_sum_exp(bias_correction);
    Posterior = sum(PointPosteriors) - weight + cauchy_lpdf(sigma_MB | 0, 1.0) + cauchy_lpdf(sigma_x1 | 0, 2.5) + cauchy_lpdf(sigma_c | 0, 2.5) + lkj_corr_cholesky_lpdf(intrinsic_correlation | 4);

}
model {
    target += Posterior;
}