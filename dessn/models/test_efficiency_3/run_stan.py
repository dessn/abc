import numpy as np
import pickle
import os

stan_code = """

data {
  real noise;          // measurement noise
  real minflux;        // minimum flux

  int N;               // Number of supernovae
  vector<lower=minflux>[N] d;          // data
}

parameters {
  real<lower=0, upper=200> mu;
  real<lower=0, upper=100> sigma;
  vector<lower=minflux, upper=200>[N] s;
}

model {
  for (n in 1:N){
    s[n] ~ normal(mu, sigma);
    d[n] ~ normal(s[n], noise) T[minflux,];
  }
}
"""


def get_data(seed=5):
    np.random.seed(seed=seed)
    mean = 100.0
    std = 20.0
    alpha = 80
    n = 1000

    actual = np.random.normal(loc=mean, scale=std, size=n)

    errors = 5
    observed = actual + np.random.normal(size=n) * errors

    mask = observed > alpha
    print(mask.sum(), n, observed[mask].mean())

    return mean, std, observed[mask], errors, alpha, mask.sum()


if __name__ == "__main__":
    import pystan

    mean, std, d, e, minFlux, nsne = get_data()
    data = {'N': nsne, 'd': d, 'noise': e, 'minflux': minFlux}
    sm = pystan.StanModel(model_code=stan_code)
    fit = sm.sampling(data=data, iter=5000, chains=4)
    dir_name = os.path.dirname(__file__)
    t = os.path.abspath(dir_name + "/output/temp.pkl")
    with open(t, 'wb') as output:
        pickle.dump(fit.extract(), output)
