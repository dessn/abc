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

    // Input ancillary data
    real <lower=0.0, upper = 1.0> p_high_mass [n_sne]; // Probability each SN is in high mass gal
    real <lower=1.0, upper = 1000.0> redshift_pre_comp [n_sne]; // Probability each SN is in high mass gal

    // Helper data used for Simpsons rule.
    real <lower=0> zs[n_z]; // List of redshifts to manually integrate over.
    int redshift_indexes[n_sne]; // Index of supernova redshifts (mapping zs -> redshifts)
}

parameters {
    ///////////////// Underlying parameters
    // Cosmology
    real <lower = 0, upper = 1> Om;
    real <lower = -2, upper = -0.4> w;
    // Supernova model
    real <lower = -20, upper = -18.> MB;
    real <lower = -0.3, upper = 0.5> alpha;
    real <lower = 0, upper = 5> beta;
    real <lower = 0, upper = 1> sigma_int;
    // Other effects
    real <lower = -0.2, upper = 0.2> dscale; // Scale of mass correction
    real <lower = 0, upper = 1> dratio; // Controls redshift dependence of correction

    ///////////////// Latent Parameters
    real <lower = -8, upper = 8> true_x1[n_sne];
    real <lower = -1, upper = 2> true_c[n_sne];

    ///////////////// Hyper Parameters
    // Colour distribution
    real <lower=-2, upper=2> c_loc;
    real <lower=0, upper=1> c_scale;
    real <lower=-5, upper=5> c_alpha;
    // Stretch distribution
    real <lower=-5, upper=5> x1_loc;
    real <lower=0, upper=5> x1_scale;

}

transformed parameters {
    // Our SALT2 model
    vector [3] model_mBx1c [n_sne];
    matrix [3,3] model_mBx1c_cov [n_sne];

    // Helper variables for Simpsons rule
    real Hinv [n_z];
    real cum_simps[n_simps];
    real model_mu[n_sne];

    // Modelling intrinsic dispersion
    vector [3] intrinsic;
    matrix [3,3] int_mat;

    // Lets actually record the proper posterior values
    vector [n_sne] PointPosteriors;

    // Other temp variables for corrections
    real mass_correction;

    // -------------Begin numerical integration-----------------
    for (i in 1:n_z) {
        Hinv[i] = 1./sqrt( Om*pow(1. + zs[i], 3) + (1. - Om) * pow(1. + zs[i], 3 * (1 + w))) ;
    }
    cum_simps[1] = 0.;
    for (i in 2:n_simps) {
        cum_simps[i] = cum_simps[i - 1] + (Hinv[2*i - 1] + 4. * Hinv[2*i - 2] + Hinv[2*i - 3])*(zs[2*i - 1] - zs[2*i - 3])/6.;
    }
    for (i in 1:n_sne) {
        model_mu[i] = 5.*log10((1. + redshifts[i])*cum_simps[redshift_indexes[i]]) + 43.158613314568356; // End is 5log10(c/H0/10pc), H0=70
    }
    // -------------End numerical integration---------------


    // Calculate intrinsic dispersion. At the moment, only considering dispersion in m_B
    intrinsic[1] = sigma_int;
    intrinsic[2] = 0.;
    intrinsic[3] = 0.;
    int_mat = intrinsic * intrinsic';

    // Now update the posterior using each supernova sample
    for (i in 1:n_sne) {
        mass_correction = dscale * (1.9 * (1 - dratio) / redshift_pre_comp[i] + dratio);

        model_mBx1c[i][1] = MB + model_mu[i] - alpha*true_x1[i] + beta*true_c[i] - mass_correction * p_high_mass[i];
        model_mBx1c[i][2] = true_x1[i];
        model_mBx1c[i][3] = true_c[i];

        // Add in intrinsic scatter
        model_mBx1c_cov[i] = obs_mBx1c_cov[i] + obs_mBx1c_cor[i] .* int_mat ;


        //if (i == 1) { print("mb=", i, " ", model_mBx1c[i][1], "|", obs_mBx1c[i][1],"  ", alpha, " ", beta, " ", MB, " ", model_mu[i], " ", true_x1[i], " ", true_c[i]); }

        PointPosteriors[i] = multi_normal_lpdf(obs_mBx1c[i] | model_mBx1c[i], model_mBx1c_cov[i]);
        PointPosteriors[i] = PointPosteriors[i] + normal_lpdf(true_x1[i] | x1_loc, x1_scale);
        PointPosteriors[i] = PointPosteriors[i] + skew_normal_lpdf(true_c[i] | c_loc, c_scale, c_alpha);
    }

}
model {
    target += sum(PointPosteriors);
    sigma_int ~ cauchy(0, 2.5);
    x1_scale ~ cauchy(0, 2.5);
    c_scale ~ cauchy(0, 2.5);
}