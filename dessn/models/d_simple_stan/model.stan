data {

    // Declaring array and data sizes
    int<lower=0> n_sne; // Number of supernovae
    int<lower=0> n_z; // Number of redshift points
    int<lower=0> n_simps; // Number of points in simpsons algorithm

    // The input summary statistics from light curve fitting
    vector[3] obs_mBx1c [n_sne]; // SALT2 fits
    matrix[3,3] obs_mBx1c_cov [n_sne]; // Covariance of SALT2 fits
    matrix[3,3] obs_mBx1c_cor [n_sne]; // Correlation of SALT2 fits

    // Input redshift data, assumed perfect redshift for spectroscopic sample
    real <lower=0> redshifts[n_sne]; // The redshift for each SN.

    // Helper data used for Simpsons rule.
    real <lower=0> zs[n_z]; // List of redshifts to manually integrate over.
    int redshift_indexes[n_sne]; // Index of supernova redshifts (mapping zs -> redshifts)
}

parameters {
    // Underlying parameters
    real <lower = -20, upper = -18.> MB;
    real <lower = 0, upper = 1> Om;
    real <lower = -2, upper = -0.4> w;
    real <lower = -0.3, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;
    real <lower = 0, upper = 1> sigma_int;

    // Latent Parameters
    real <lower = -8, upper = 8> true_x1[n_sne];
    real <lower = -1, upper = 2> true_c[n_sne];

    // Hyper Parameters



}

transformed parameters {

    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];

    vector [3] intrinsic;
    matrix [3,3] int_mat;
    vector [n_sne] PointPosteriors;

    // -------------Begin numerical integration-----------------
    for (i in 1:n_z) {
        Hinv[i] <- 1./sqrt( Om*pow(1. + zs[i], 3) + (1. - Om) * pow(1. + zs[i], 3 * (1 + w))) ;
    }
    cum_simps[1] <- 0.; // Redshift = 0 should be first element!
    for (i in 2:n_simps) {
        cum_simps[i] <- cum_simps[i - 1] + (Hinv[2*i - 1] + 4. * Hinv[2*i - 2] + Hinv[2*i - 3])*(zs[2*i - 1] - zs[2*i - 3])/6.;
    }
    for (i in 1:n_sne) {
        model_mu[i] <- 5.*log10((1. + redshifts[i])*cum_simps[redshift_indexes[i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
    }
    // -------------End numerical integration---------------

    intrinsic[1] <- sigma_int;
    intrinsic[2] <- 0.;
    intrinsic[3] <- 0.;

    int_mat <- intrinsic * intrinsic';

    for (i in 1:n_sne) {
        model_mBx1c[i][1] <- MB + model_mu[i] - alpha*true_x1[i] + beta*true_c[i];
        model_mBx1c[i][2] <- true_x1[i];
        model_mBx1c[i][3] <- true_c[i];

        // Add in intrinsic scatter
        model_mBx1c_cov[i] <- obs_mBx1c_cov[i] + obs_mBx1c_cor[i] .* int_mat ;


        //if (i == 1) { print("mb=", i, " ", model_mBx1c[i][1], "|", obs_mBx1c[i][1],"  ", alpha, " ", beta, " ", MB, " ", model_mu[i], " ", true_x1[i], " ", true_c[i]); }

        PointPosteriors[i] <- multi_normal_log(obs_mBx1c[i], model_mBx1c[i], model_mBx1c_cov[i])
            + normal_log(true_x1[i], 0, 1)
            + normal_log(true_c[i], 0, 0.1);
    }

}
model {
    increment_log_prob(sum(PointPosteriors));
    sigma_int ~ cauchy(0, 2.5);
}