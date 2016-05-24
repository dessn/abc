stan_code = """

data {
  real noise;          // measurement noise
  real minflux;        // minimum flux

  int N;               // Number of supernovae
  vector<lower=minflux> d[N];          // data
}

parameters {
  real<lower=0, upper=200> mu;
  real<lower=0, upper=100> sigma;
  vector s[N];
}

model {
  for (n in 1:N){
    s[n] ~ normal(mu[n], sigma);
    d[n] ~ normal(s[n], noise) T[minflux,];
  }
}
"""

data = {'N': nsne, 'd': 4, 'noise': 2, 'minflux': color_renorm}
sm = pystan.StanModel(model_code=stan_code)
fit = sm.sampling(data=data, iter=10000, chains=4)

output = open('temp.pkl','wb')
pickle.dump(fit.extract(), output)
output.close()