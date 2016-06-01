import os
import pickle

from dessn.proofs.efficiency_3.efficiency_model_3 import get_data

stan_code = """

data {
  real noise;          // measurement noise

  int N;               // Number of supernovae
  vector[N] d;          // data
}

parameters {
  real<lower=0, upper=200> mu;
  real<lower=0, upper=100> sigma;
  vector<lower=0, upper=200>[N] s;
}

model {
  for (n in 1:N){
    s[n] ~ normal(mu, sigma);
    d[n] ~ normal(s[n], noise);
  }
}
"""


if __name__ == "__main__":
    import pystan
    for i in range(10):
        mean, std, observed, errors, alpha, actual, uo, oe = get_data(seed=100 * (i + 1))
        data = {'N': uo.size, 'd': uo, 'noise': oe[0]}
        sm = pystan.StanModel(model_code=stan_code)
        fit = sm.sampling(data=data, iter=10000, chains=4)
        dir_name = os.path.dirname(__file__)
        t = os.path.abspath(dir_name + "/output/temp%d.pkl" % i)
        with open(t, 'wb') as output:
            pickle.dump(fit.extract(), output)