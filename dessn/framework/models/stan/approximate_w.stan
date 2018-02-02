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
    //real <lower = -0.2, upper = 0.2> delta_alpha;
    real <lower = 0, upper = 5> beta;
    //real <lower = -2, upper = 2> delta_beta;

    // Other effects
    real <lower = -0.2, upper = 0.4> dscale; // Scale of mass correction
    real <lower = 0, upper = 1> dratio; // Controls redshift dependence of correction
    vector[n_calib] calibration;

    ///////////////// Latent Parameters
    vector[3] deviations [n_sne];
    vector[4] deltas [n_surveys];

    ///////////////// Population (Hyper) Parameters
    real <lower = -20.5, upper = -18.5> mean_MB;
    matrix <lower = -2.0, upper = 2.0> [n_surveys, num_nodes] mean_x1;
    matrix <lower = -0.3, upper = 0.3> [n_surveys, num_nodes] mean_c;
    real <lower = -4, upper = -0.5> log_sigma_MB [n_surveys];
    real <lower = -6, upper = 1> log_sigma_x1 [n_surveys];
    real <lower = -8, upper = -1.0> log_sigma_c [n_surveys];
    real <lower = 0, upper = 0.98> delta_c [n_surveys];
    real<lower=0, upper=1.0>  kappa_c0 [n_surveys];
    real<lower=0, upper=10>  kappa_c1 [n_surveys];
    real<lower=-1, upper=1>  rho_c [n_surveys];

}

transformed parameters {
    cholesky_factor_corr[3] intrinsic_correlation [n_surveys];

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

    // Unexplained dispersion
    vector[3] diag_extra [n_sne];

    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];

    // Modelling intrinsic dispersion
    matrix [3,3] population [n_surveys];
    matrix [3,3] full_sigma [n_surveys];
    vector [3] mean_MBx1c [n_sne];
    vector [3] mean_MBx1c_out [n_sne];

    // Variables to calculate the bias correction
    real mB_mean [n_surveys];
    real mB_width [n_surveys];
    real mB_alpha [n_surveys];
    real mB_norm [n_surveys];
    real mB_width2 [n_surveys];
    real mB_alpha2 [n_surveys];

    vector[4] shifts [n_surveys];
    real cor_mB_mean [n_sne];
    real cor_sigma [n_surveys];
    real cor_mb_width2 [n_surveys];
    real cor_mb_norm_width [n_surveys];
    real mean_c_adjust [n_surveys];
    real sigma_c_adjust [n_surveys];
    real sigma_c_adjust_ratio [n_surveys];
    real alpha_c [n_surveys];

    real cor_mB_mean_out [n_sne];
    real cor_sigma_out [n_surveys];
    real cor_mb_width2_out;
    real cor_mb_norm_width_out [n_surveys];

    // Lets actually record the proper posterior values
    vector [n_sne] point_posteriors;
    vector [n_surveys] survey_posteriors;
    vector [n_sne] weights;
    vector [n_sne] numerator_weight;
    real weight;
    real posterior;

    // Other temp variables for corrections
    real mass_correction;

    // -------------Begin numerical integration-----------------
    real expon;
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

    // Calculate intrinsic dispersion and selection effects for each survey
    cor_mb_width2_out = outlier_population[1,1]^2 + (alpha * outlier_population[2,2])^2 + (beta * outlier_population[3,3])^2;


    for (i in 1:n_surveys) {

        intrinsic_correlation[i] = diag_matrix(rep_vector(1, 3));

        shifts[i] = mb_cov_chol[i] * deltas[i] * systematics_scale;
        mB_mean[i] = mB_mean_orig[i] + shifts[i][1];
        mB_width[i] = mB_width_orig[i] + shifts[i][2];
        mB_alpha[i] = mB_alpha_orig[i] + shifts[i][3];
        mB_norm[i] = log(mB_norm_orig[i] + shifts[i][4]);
        mB_alpha2[i] = mB_alpha[i]^2;
        mB_width2[i] = mB_width[i]^2;


        // Move from log space back to real space
        sigma_MB[i] = exp(log_sigma_MB[i]);
        sigma_x1[i] = exp(log_sigma_x1[i]);
        sigma_c[i] = exp(log_sigma_c[i]);

        alpha_c[i] = delta_c[i] / sqrt(1 - delta_c[i]^2);
        mean_c_adjust[i] = frac_shift * delta_c[i] * sqrt(2 / pi()) * sigma_c[i];
        sigma_c_adjust_ratio[i] = sqrt(1 - (2 * delta_c[i]^2 / pi()));
        sigma_c_adjust[i] = 1 + (frac_shift * (sigma_c_adjust_ratio[i] - 1));

        // Calculate selection effect widths
        cor_mb_width2[i] = sigma_MB[i]^2 + (alpha * sigma_x1[i])^2 + (beta * sigma_c[i] * sigma_c_adjust[i])^2;
        cor_sigma[i] = sqrt(((cor_mb_width2[i] + mB_width2[i]) / mB_width2[i])^2 * ((mB_width2[i] / mB_alpha2[i]) + ((mB_width2[i] * cor_mb_width2[i]) / (cor_mb_width2[i] + mB_width2[i]))));

        cor_mb_norm_width[i] = sqrt(mB_width2[i] + cor_mb_width2[i]);

        cor_sigma_out[i] = sqrt(((cor_mb_width2_out + mB_width2[i]) / mB_width2[i])^2 * ((mB_width2[i] / mB_alpha2[i]) + ((mB_width2[i] * cor_mb_width2_out) / (cor_mb_width2_out + mB_width2[i]))));
        cor_mb_norm_width_out[i] = sqrt(mB_width2[i] + cor_mb_width2_out);

    }

    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {

        // Intrinsic
        diag_extra[i][1] = 0;
        diag_extra[i][2] = 0;
        diag_extra[i][3] = sigma_MB[survey_map[i]] * kappa_c0[survey_map[i]] * (1 + kappa_c1[survey_map[i]] * redshifts[i]);

        // redshift dependent effects
        mean_x1_sn[i] = dot_product(mean_x1[survey_map[i]], node_weights[i]);
        mean_c_sn[i] = dot_product(mean_c[survey_map[i]], node_weights[i]);

        mean_MBx1c[i][1] = mean_MB;
        mean_MBx1c[i][2] = mean_x1_sn[i];
        mean_MBx1c[i][3] = mean_c_sn[i];

        mean_MBx1c_out[i][1] = mean_MB - outlier_MB_delta;
        mean_MBx1c_out[i][2] = mean_x1_sn[i];
        mean_MBx1c_out[i][3] = mean_c_sn[i];

        // Calculate mass correction
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        // Convert into apparent magnitude
        model_mBx1c[i] = obs_mBx1c[i] + (obs_mBx1c_chol[i] + diag_matrix(diag_extra[i])) * deviations[i];

        // Add calibration uncertainty
        model_mBx1c[i] = model_mBx1c[i] + deta_dcalib[i] * calibration * systematics_scale;

        // Convert population into absolute magnitude
        model_MBx1c[i][1] = model_mBx1c[i][1] - model_mu[i] + alpha * model_mBx1c[i][2] - beta * model_mBx1c[i][3] + mass_correction * masses[i];
        model_MBx1c[i][2] = model_mBx1c[i][2];
        model_MBx1c[i][3] = model_mBx1c[i][3];

        // Mean of population
        cor_mB_mean[i] = mean_MB + model_mu[i] - alpha * mean_x1_sn[i] + beta * (mean_c_sn[i] + mean_c_adjust[survey_map[i]]) - mass_correction * masses[i];
        cor_mB_mean_out[i] = cor_mB_mean[i] - outlier_MB_delta;

        if (correction_skewnorm[survey_map[i]]) {
            weights[i] = log_sum_exp(
                log(prob_ia[i]) + mB_norm[survey_map[i]] + normal_lpdf(cor_mB_mean[i] | mB_mean[survey_map[i]], cor_mb_norm_width[survey_map[i]]) + normal_lcdf(mB_sgn_alpha[survey_map[i]] * (cor_mB_mean[i] - mB_mean[survey_map[i]])| 0, cor_sigma[survey_map[i]]),
                log(1 - prob_ia[i]) + mB_norm[survey_map[i]] + normal_lpdf(cor_mB_mean_out[i] | mB_mean[survey_map[i]], cor_mb_norm_width_out[survey_map[i]]) + normal_lcdf(mB_sgn_alpha[survey_map[i]] * (cor_mB_mean_out[i] - mB_mean[survey_map[i]])| 0, cor_sigma_out[survey_map[i]])
            );
            numerator_weight[i] = mB_norm[survey_map[i]] + skew_normal_lpdf(model_mBx1c[i][1] | mB_mean[survey_map[i]], mB_width[survey_map[i]], mB_alpha[survey_map[i]]);
        } else {
            weights[i] = log_sum_exp(
                log(prob_ia[i]) + mB_norm[survey_map[i]] + normal_lccdf(cor_mB_mean[i] | mB_mean[survey_map[i]], cor_mb_norm_width[survey_map[i]]),
                log(1 - prob_ia[i]) + mB_norm[survey_map[i]] + normal_lccdf(cor_mB_mean_out[i] | mB_mean[survey_map[i]], cor_mb_norm_width_out[survey_map[i]])
            );
            numerator_weight[i] = log_sum_exp(-10, mB_norm[survey_map[i]] + normal_lccdf(model_mBx1c[i][1] | mB_mean[survey_map[i]], mB_width[survey_map[i]]));
        }
        // Track and update posterior
        point_posteriors[i] = normal_lpdf(deviations[i] | 0, 1)
            + log_sum_exp(
                log(prob_ia[i]) + normal_lpdf(model_MBx1c[i][1] | mean_MB, sqrt(sigma_MB[survey_map[i]]^2 + rho_c[survey_map[i]] * diag_extra[i][3]^2))
                + normal_lpdf(model_MBx1c[i][2] | mean_x1_sn[i], sigma_x1[survey_map[i]])
                + skew_normal_lpdf(model_MBx1c[i][3] | mean_c_sn[i], sigma_c[survey_map[i]], alpha_c[survey_map[i]]),
                log(1 - prob_ia[i]) + multi_normal_cholesky_lpdf(model_MBx1c[i] | mean_MBx1c_out[i], outlier_dispersion))
            + numerator_weight[i];
    }
    weight = sum(weights);
    for (i in 1:n_surveys) {
        survey_posteriors[i] = normal_lpdf(mean_x1[i]  | 0, 1.0)
            + normal_lpdf(mean_c[i]  | 0, 0.1)
            + normal_lpdf(deltas[i] | 0, 1)
            + cauchy_lpdf(kappa_c0[i] | 0, 0.5)
            + cauchy_lpdf(kappa_c1[i] | 0, 3)
            + lkj_corr_cholesky_lpdf(intrinsic_correlation[i] | 4);
    }
    posterior = sum(point_posteriors) + sum(survey_posteriors)
        + cauchy_lpdf(sigma_MB | 0, 1)
        + cauchy_lpdf(sigma_x1 | 0, 1)
        + cauchy_lpdf(sigma_c  | 0, 1)
        + normal_lpdf(calibration | 0, 1);
}
model {
    target += posterior;
    
    if (apply_efficiency) {
        target += -weight;
    }
    
    if (apply_prior) {
        target += normal_lpdf(Om | 0.3, 0.01);
    }
    target += normal_lpdf(mean_MB | -19.365, 0.01);
}