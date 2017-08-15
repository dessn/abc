data {

    // Declaring array and data sizes
    int<lower=0> n_sne; // Number of supernovae
    int<lower=0> n_z; // Number of redshift points
    int<lower=0> n_simps; // Number of points in simpsons algorithm

    int<lower=0>  n_surveys; // How many surveys we are analysing
    int<lower=0>  n_snes [n_surveys]; // How many surveys we are analysing
    int<lower=0>  survey_map [n_sne]; // A bit from supernova to survey
    int<lower=0>  n_calib; // How many calibration

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
    real mB_mean [n_surveys];
    real mB_width [n_surveys];
    real mB_alpha [n_surveys];
    real mB_sgn_alpha [n_surveys];
    real mean_mass [n_surveys];
    real mB_norms [n_surveys];
    int correction_skewnorm [n_surveys];

    // Redshift grid to do integral
    int<lower=0> n_sim; // Number of added redshift points
    real <lower=0> sim_redshifts[n_surveys, n_sim]; // The redshift values
    int sim_redshift_indexes[n_surveys, n_sim]; // Index of added redshifts (mapping zs -> redshifts)
    real sim_log_weight[n_surveys, n_sim]; // The weights to use for simpsons rule multipled by the probability P(z) for each redshift
    vector[num_nodes] sim_node_weights [n_surveys, n_sim]; // Each supernova's node weight
    real <lower=1.0, upper = 1000.0> sim_redshift_pre_comp [n_surveys, n_sim]; // Precomputed function of redshift for speed for sim redshifts

    // Calibration std
    matrix[3, n_calib] deta_dcalib [n_sne]; // Sensitivity of summary stats to change in calib
}
transformed data {
    matrix[3, 3] obs_mBx1c_chol [n_sne];
    real mB_width2 [n_surveys];
    real mB_alpha2 [n_surveys];

    for (i in 1:n_sne) {
        obs_mBx1c_chol[i] = cholesky_decompose(obs_mBx1c_cov[i]);
    }
    for (i in 1:n_surveys) {
        mB_width2[i] = mB_width[i]^2;
        mB_alpha2[i] = mB_alpha[i]^2;
    }
}

parameters {
    ///////////////// Underlying parameters
    // Cosmology
    real <lower = 0.05, upper = 0.99> Om;
    // real <lower = -2, upper = -0.4> w;
    // Supernova model
    real <lower = -0.2, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;

    // Other effects
    real <lower = -0.2, upper = 0.4> dscale; // Scale of mass correction
    real <lower = 0, upper = 1> dratio; // Controls redshift dependence of correction
    vector[n_calib] calibration;

    ///////////////// Latent Parameters
    vector[3] deviations [n_sne];

    ///////////////// Population (Hyper) Parameters
    real <lower = -21, upper = -18> mean_MB;
    matrix <lower = -1.0, upper = 1.0> [n_surveys, num_nodes] mean_x1;
    matrix <lower = -0.2, upper = 0.2> [n_surveys, num_nodes] mean_c;
    real <lower = -4, upper = 1> log_sigma_MB [n_surveys];
    real <lower = -4, upper = 1> log_sigma_x1 [n_surveys];
    real <lower = -4, upper = 1> log_sigma_c [n_surveys];
    cholesky_factor_corr[3] intrinsic_correlation [n_surveys];

}

transformed parameters {

    // Back to real space
    real sigma_MB [n_surveys];
    real sigma_x1 [n_surveys];
    real sigma_c [n_surveys];

    // Pop at given redshift
    real mean_x1_sn [n_sne];
    real mean_c_sn [n_sne];

    // Our SALT2 model
    vector [3] model_MBx1c [n_sne];
    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];

    // Modelling intrinsic dispersion
    matrix [3,3] population [n_surveys];
    matrix [3,3] full_sigma [n_surveys];
    vector [3] mean_MBx1c [n_sne];
    vector [3] sigmas [n_surveys];

    // Variables to calculate the bias correction
    real sim_model_mu[n_surveys, n_sim];
    real cor_MB_mean [n_surveys, n_sim];
    real cor_mB_mean [n_surveys, n_sim];
    real cor_sigma [n_surveys];
    real cor_mb_width2 [n_surveys];
    real cor_mb_norm_width [n_surveys];
    real cor_x1_val[n_surveys, n_sim];
    real cor_c_val[n_surveys, n_sim];
    vector[n_sim] cor_mB_cor[n_surveys];
    vector[n_sim] cor_mB_cor_weighted[n_surveys];

    // Lets actually record the proper posterior values
    vector [n_sne] point_posteriors;
    vector [n_surveys] survey_posteriors;
    vector [n_surveys] weights;
    vector [n_sne] numerator_weight;
    real weight;
    real posterior;

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
    for (j in 1:n_surveys) {
        for (i in 1:n_sim) {
            sim_model_mu[j,i] = 5.*log10((1. + sim_redshifts[j,i])*cum_simps[sim_redshift_indexes[j,i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
        }
    }
    // -------------End numerical integration---------------

    // Calculate intrinsic dispersion and selection effects for each survey
    for (i in 1:n_surveys) {
        // Move from log space back to real space
        sigma_MB[i] = exp(log_sigma_MB[i]);
        sigma_x1[i] = exp(log_sigma_x1[i]);
        sigma_c[i] = exp(log_sigma_c[i]);

        // Construct sigma vector
        sigmas[i][1] = sigma_MB[i];
        sigmas[i][2] = sigma_x1[i];
        sigmas[i][3] = sigma_c[i];

        // Turn this into population matrix
        population[i] = diag_pre_multiply(sigmas[i], intrinsic_correlation[i]);
        full_sigma[i] = population[i] * population[i]';

        // Calculate selection effect widths
        cor_mb_width2[i] = sigma_MB[i]^2 + (alpha * sigma_x1[i])^2 + (beta * sigma_c[i])^2 + 2 * (-alpha * full_sigma[i][1][2] + beta * (full_sigma[i][1][3]) - alpha * beta * (full_sigma[i][2][3] ));
        cor_sigma[i] = sqrt(((cor_mb_width2[i] + mB_width2[i]) / mB_width2[i])^2 * ((mB_width2[i] / mB_alpha2[i]) + ((mB_width2[i] * cor_mb_width2[i]) / (cor_mb_width2[i] + mB_width2[i]))));

        cor_mb_norm_width[i] = sqrt(mB_width2[i] + cor_mb_width2[i]);
    }

    // Here I do another simpsons rule, but in log space. So each f(x) is in log space, the weights are log'd
    // and we add in log_sum_exp
    for (j in 1:n_surveys) {
        if (correction_skewnorm[j]) {
            for (i in 1:n_sim) {
                mass_correction = dscale * (1.9 * (1 - dratio) / sim_redshift_pre_comp[j][i] + dratio);
                cor_x1_val[j,i] = dot_product(mean_x1[j], sim_node_weights[j,i]);
                cor_c_val[j,i] = dot_product(mean_c[j], sim_node_weights[j,i]);
                cor_mB_mean[j,i] = mean_MB - alpha*cor_x1_val[j,i] + beta*cor_c_val[j,i] + sim_model_mu[j,i] - mass_correction * mean_mass[j];
                cor_mB_cor[j][i] = log(2) + mB_norms[j] + normal_lpdf(cor_mB_mean[j,i] | mB_mean[j], cor_mb_norm_width[j]) + normal_lcdf(mB_sgn_alpha[j] * (cor_mB_mean[j,i] - mB_mean[j]) | 0, cor_sigma[j]);
                cor_mB_cor_weighted[j][i] = cor_mB_cor[j][i] + sim_log_weight[j,i];
            }
        } else {
            for (i in 1:n_sim) {
                mass_correction = dscale * (1.9 * (1 - dratio) / sim_redshift_pre_comp[j][i] + dratio);
                cor_x1_val[j,i] = dot_product(mean_x1[j], sim_node_weights[j,i]);
                cor_c_val[j,i] = dot_product(mean_c[j], sim_node_weights[j,i]);
                cor_mB_mean[j,i] = mean_MB - alpha*cor_x1_val[j,i] + beta*cor_c_val[j,i] + sim_model_mu[j,i] - mass_correction * mean_mass[j];
                cor_mB_cor[j][i] = normal_lccdf(cor_mB_mean[j,i] | mB_mean[j], cor_mb_norm_width[j]);
                cor_mB_cor_weighted[j][i] = cor_mB_cor[j][i] + sim_log_weight[j,i];
            }
        }
        weights[j] = n_snes[j] * log_sum_exp(cor_mB_cor_weighted[j]);
    }


    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {

        // redshift dependent effects
        mean_x1_sn[i] = dot_product(mean_x1[survey_map[i]], node_weights[i]);
        mean_c_sn[i] = dot_product(mean_c[survey_map[i]], node_weights[i]);

        mean_MBx1c[i][1] = mean_MB;
        mean_MBx1c[i][2] = mean_x1_sn[i];
        mean_MBx1c[i][3] = mean_c_sn[i];

        // Calculate mass correction
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        // Convert into apparent magnitude
        model_mBx1c[i] = obs_mBx1c[i] + obs_mBx1c_chol[i] * deviations[i];

        // Add calibration uncertainty
        // model_mBx1c[i] = model_mBx1c[i] + deta_dcalib[i] * calibration;

        // Convert population into absolute magnitude
        model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha*model_mBx1c[i][2] - beta*model_mBx1c[i][3] + mass_correction * masses[i];
        model_MBx1c[i][2] = model_mBx1c[i][2];
        model_MBx1c[i][3] = model_mBx1c[i][3];

        if (correction_skewnorm[survey_map[i]]) {
            numerator_weight[i] = skew_normal_lpdf(model_mBx1c[i][1] | mB_mean[survey_map[i]], mB_width[survey_map[i]], mB_alpha[survey_map[i]]);
        } else {
            numerator_weight[i] = normal_lccdf(model_mBx1c[i][1] | mB_mean[survey_map[i]], mB_width[survey_map[i]]);
        }

        // Track and update posterior
        point_posteriors[i] = normal_lpdf(deviations[i] | 0, 1)
            + multi_normal_cholesky_lpdf(model_MBx1c[i] | mean_MBx1c[i], population[survey_map[i]]);
            // + numerator_weight[i];
    }
    weight = sum(weights);
    for (i in 1:n_surveys) {
        survey_posteriors[i] = normal_lpdf(mean_x1[i]  | 0, 1)
            + normal_lpdf(mean_c[i]  | 0, 0.1)
            + lkj_corr_cholesky_lpdf(intrinsic_correlation[i] | 4);
    }
    posterior = sum(point_posteriors) - weight + sum(survey_posteriors)
        + cauchy_lpdf(sigma_MB | 0, 2.5)
        + cauchy_lpdf(sigma_x1 | 0, 2.5)
        + cauchy_lpdf(sigma_c  | 0, 2.5)
        + normal_lpdf(beta | 0, 0.01)
        + normal_lpdf(alpha | 0, 0.01)
        + normal_lpdf(calibration | 0, 1);
}
model {
    target += posterior;
}