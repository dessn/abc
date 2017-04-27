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

    // Data for redshift nodes
    int num_nodes; // Num redshift nodes
    vector[num_nodes] node_weights [n_sne]; // Each supernova's node weight

    // Approximate correction in mB
    real mB_mean;
    real mB_width2;
    real mB_alpha2;
    real mean_mass;

    // Redshift grid to do integral
    int<lower=0> n_sim; // Number of added redshift points
    real <lower=0> sim_redshifts[n_sim]; // The redshift values
    int sim_redshift_indexes[n_sim]; // Index of added redshifts (mapping zs -> redshifts)
    real sim_log_weight[n_sim]; // The weights to use for simpsons rule multipled by the probability P(z) for each redshift
    vector[num_nodes] sim_node_weights [n_sim]; // Each supernova's node weight
    real <lower=1.0, upper = 1000.0> sim_redshift_pre_comp [n_sim]; // Precomputed function of redshift for speed for sim redshifts



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
    vector <lower = -1.0, upper = 1.0> [num_nodes] mean_x1;
    vector <lower = -0.2, upper = 0.2> [num_nodes] mean_c;
    real <lower = -10, upper = 1> log_sigma_MB;
    real <lower = -10, upper = 1> log_sigma_x1;
    real <lower = -10, upper = 1> log_sigma_c;
    cholesky_factor_corr[3] intrinsic_correlation;

}

transformed parameters {
    real mB_width;
    real mB_alpha;

    // Back to real space
    real sigma_MB;
    real sigma_x1;
    real sigma_c;

    // Our SALT2 model
    vector [3] model_MBx1c [n_sne];
    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];

    // Modelling intrinsic dispersion
    matrix [3,3] population;
    matrix [3,3] full_sigma;
    vector [3] mean_MBx1c [n_sne];
    vector [3] sigmas;

    // Variables to calculate the bias correction
    real sim_model_mu[n_sim];
    real cor_MB_mean;
    real cor_mb_width2;
    real cor_mB_mean [n_sim];
    real cor_x1_val [n_sim];
    real cor_c_val [n_sim];
    real cor_mB_cor [n_sim];
    real cor_mB_cor_weighted [n_sim];
    real cor_sigma2;

    // Lets actually record the proper posterior values
    vector [n_sne] PointPosteriors;
    real weight;
    real posterior;

    // Other temp variables for corrections
    real mass_correction;

    // Correct for colour skewness
    real delta_c;
    real mean_c_skew;
    real sigma_c_skew;

    mB_width = sqrt(mB_width2);
    mB_alpha = sqrt(mB_alpha2);

    sigma_MB = exp(log_sigma_MB);
    sigma_x1 = exp(log_sigma_x1);
    sigma_c = exp(log_sigma_c);

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

    // Calculate intrinsic dispersion. At the moment, only considering dispersion in m_B
    sigmas[1] = sigma_MB;
    sigmas[2] = sigma_x1;
    sigmas[3] = sigma_c;
    population = diag_pre_multiply(sigmas, intrinsic_correlation);
    full_sigma = population * population';

    // Calculate mean pop
    cor_mb_width2 = sigma_MB^2 + (alpha * sigma_x1)^2 + (beta * sigma_c)^2 + 2 * (-alpha * full_sigma[1][2] + beta * (full_sigma[1][3]) - alpha * beta * (full_sigma[2][3] ));
    cor_sigma2 = ((cor_mb_width2 + mB_width2) / mB_width2)^2 * ((mB_width2 / mB_alpha2) + ((mB_width2 * cor_mb_width2) / (cor_mb_width2 + mB_width2)));


    // Here I do another simpsons rule, but in log space. So each f(x) is in log space, the weights are log'd
    // and we add in log_sum_exp
    for (i in 1:n_sim) {
        mass_correction = dscale * (1.9 * (1 - dratio) / sim_redshift_pre_comp[i] + dratio);
        cor_x1_val[i] = dot_product(mean_x1, sim_node_weights[i]);
        cor_c_val[i] = dot_product(mean_c, sim_node_weights[i]);
        cor_mB_mean[i] = mean_MB - alpha* cor_x1_val[i] + beta*cor_c_val[i] + sim_model_mu[i];// - mass_correction * mean_mass;
        cor_mB_cor[i] = normal_lpdf(mB_mean | cor_mB_mean[i], sqrt(mB_width2 + cor_mb_width2)) + normal_lcdf(cor_mB_mean[i] | mB_mean, sqrt(cor_sigma2));
        cor_mB_cor_weighted[i] = cor_mB_cor[i] + sim_log_weight[i];
    }
    //weight = n_sne * log_sum_exp(cor_mB_cor_weighted);
    weight = 0;
    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {

        // redshift dependent effects
        mean_MBx1c[i][1] = mean_MB;
        mean_MBx1c[i][2] = dot_product(mean_x1, node_weights[i]);
        mean_MBx1c[i][3] = dot_product(mean_c, node_weights[i]);

        // Calculate mass correction
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        // Convert into apparent magnitude
        model_mBx1c[i] = obs_mBx1c[i] + obs_mBx1c_chol[i] * deviations[i];

        // Add calibration uncertainty
        //model_mBx1c[i] = model_mBx1c[i] + deta_dcalib[i] * (calib_std .* calibration);

        // Convert population into absolute magnitude
        model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha*model_mBx1c[i][2] - beta*model_mBx1c[i][3];// + mass_correction * masses[i];
        model_MBx1c[i][2] = model_mBx1c[i][2];
        model_MBx1c[i][3] = model_mBx1c[i][3];

        // Track and update posterior
        PointPosteriors[i] = normal_lpdf(deviations[i] | 0, 1)
            + multi_normal_cholesky_lpdf(model_MBx1c[i] | mean_MBx1c[i], population)
            + skew_normal_lpdf(model_mBx1c[i][1] | mB_mean, mB_width, mB_alpha);
    }
    posterior = sum(PointPosteriors) - weight
        + cauchy_lpdf(sigma_MB | 0, 2.5)
        + cauchy_lpdf(sigma_x1 | 0, 2.5)
        + cauchy_lpdf(sigma_c  | 0, 2.5)
        + normal_lpdf(mean_x1  | 0, 1)
        + normal_lpdf(mean_c  | 0, 0.1)
        + lkj_corr_cholesky_lpdf(intrinsic_correlation | 4)
        + normal_lpdf(calibration | 0, 1);

}
model {
    target += posterior;
}