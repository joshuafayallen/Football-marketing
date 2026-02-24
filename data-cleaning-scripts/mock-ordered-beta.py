
import polars as pl 
import pandas as pd 
import pymc as pm 
import numpy as np
import pytensor.tensor as pt
import arviz as az

ordered_beta_dat = (
    pl.read_csv("ordered_beta_dat.csv")
    .with_columns(
        ((pl.col('therm') - pl.col("therm").min())/(pl.col('therm').max() - pl.col('therm').min()))
    ))


unique_regions = ordered_beta_dat.select(pl.col('region').unique()).sort('region').to_pandas()

unique_income = ordered_beta_dat.select(pl.col('income').unique()).sort('income').to_pandas()

unique_education = ordered_beta_dat.select(pl.col('education').unique()).sort('education').to_pandas()

region_idx = pd.Categorical(
    ordered_beta_dat['relig'].to_pandas(),
    categories = unique_regions['region']
).codes

income_idx = pd.Categorical(
    ordered_beta_dat['income'].to_pandas(),
    categories=unique_income['income']
).codes

education_idx = pd.Categorical(
    ordered_beta_dat['education'].to_pandas(),
    categories=unique_education['education']
).codes

ordered_beta_use = ordered_beta_dat.to_pandas()

therm = ordered_beta_use['therm']
prop_mask = (therm > 0) & (therm < 1)
degen_mask = (therm == 0) | (therm == 1)

therm_prop = therm[prop_mask]
therm_degen = therm[degen_mask]
degen_is_zero = (therm_degen == 0)

n_prop = prop_mask.sum()
n_degen = degen_mask.sum()

d_education = len(unique_education) - 1 
d_income = len(unique_income) - 1

coords = {
    'regions':                unique_regions['region'].tolist(),
    'income':                 unique_income['income'].tolist(),
    'education':              unique_education['education'].tolist(),
    'education_simplex':      unique_education['education'].tolist()[:-1],
    'income_simplex':         unique_income['income'].tolist()[:-1],
    'education_int_simplex':  unique_education['education'].tolist()[:-1],
    'income_int_simplex':     unique_income['income'].tolist()[:-1],
    'obs_id':                 ordered_beta_use.index,
}


def monotonic_effect(name, dim_name, n_levels, obs_idx):
    D           = n_levels - 1
    simplex_dim = f"{dim_name}_simplex"  # length D, not n_levels

    b    = pm.Normal(f"b_{name}", mu=0, sigma=5)
    zeta = pm.Dirichlet(f"zeta_{name}", a=np.ones(D), dims=simplex_dim)

    cs = pt.concatenate([[0.0], pt.cumsum(zeta)])  # shape (n_levels,)
    nu = b * D * cs                                 # shape (n_levels,)

    return nu[obs_idx], nu



with pm.Model(coords = coords) as mock_orderd_beta:

    
    regions_data = pm.Data('regions_data', region_idx, dims = 'obs_id')
    education_data = pm.Data("education_data", education_idx, dims = 'obs_id')
    income_data = pm.Data('income_data', income_idx, dims = 'obs_id')

    
    c0 = pm.Normal("cutpoint_0", mu = 0, sigma = 5)
    gap = pm.Normal('cutpoint_gap', mu = 0, sigma = 3)
    c1 = pm.Deterministic('cutpoint_1', c0 + pt.exp(gap))
    
    intercept = pm.Normal('intercept', mu = 0, sigma = 1)

    edu_contrib, nu_edu = monotonic_effect('edu', 'education', len(coords['education']), education_data)
    
    income_contrib, nu_income = monotonic_effect('income', 'income', len(coords['income']), income_data) 

    b_edu_income = pm.Normal("b_edu_income", mu=0, sigma=2)

    _, nu_edu_int    = monotonic_effect(
        "edu_int",    "education", len(coords["education"]), education_data
    )
    _, nu_income_int = monotonic_effect(
        "income_int", "income",    len(coords["income"]),  income_data
    )

    interaction = (
        b_edu_income
        * (nu_edu_int[education_data]    / pt.max(nu_edu_int))
        * (nu_income_int[income_data]    / pt.max(nu_income_int))
    )

    # -- Random region intercepts  (1|region) ----------------------------
    sigma_region = pm.Exponential('region_sigma', 1)

# With non-centred parameterisation:
    region_offset_raw = pm.Normal("region_offset_raw", mu=0, sigma=1, dims="regions")
    region_offset = pm.Deterministic(
    "region_offset", region_offset_raw * sigma_region, dims="regions"
)

    # -- Linear predictor ------------------------------------------------
    eta = (
        intercept
        + edu_contrib           # education main effect
        + income_contrib        # income main effect
        + interaction           # education × income
        + region_offset[regions_data]
    )

    eta_prop  = eta[np.array(prop_mask)]
    eta_degen = eta[np.array(degen_mask)]

    log_phi = pm.Normal("log_phi", mu=0, sigma=3)
    phi = pt.exp(log_phi)

    # -- Likelihood: degenerate outcomes (exactly 0 or 1) ----------------
    # Pr(Y = 0) = 1 − σ(η − c0)   →   log prob = −softplus(η − c0)
    # Pr(Y = 1) = σ(η − c1)       →   log prob =  log_sigmoid(η − c1)
    logp_degen_0 = -pt.softplus(eta_degen - c0)
    logp_degen_1 =  -pt.softplus(-(eta_degen - c1))
    logp_degen = pt.where(
        pt.as_tensor_variable(degen_is_zero),
        logp_degen_0,
        logp_degen_1,
    )
    pm.Potential("loglik_degen", pt.sum(logp_degen))

    # -- Likelihood: proportion outcomes (strictly between 0 and 1) ------
    # Two contributions per observation:
    #   (a) The probability that this observation falls in the (0,1) bin:
    #         Pr(c0 < η* < c1) = σ(η − c0) − σ(η − c1)
    #   (b) The Beta density at the observed value given the predicted mean:
    #         BetaProportion(y | μ, φ) = Beta(y | μφ, (1−μ)φ)
    mu_prop   = pm.math.sigmoid(eta_prop)
    log_p_mid = pt.log(
        pm.math.sigmoid(eta_prop - c0) - pm.math.sigmoid(eta_prop - c1)
    )
    logp_beta = pm.logp(
        pm.Beta.dist(alpha=mu_prop * phi, beta=(1.0 - mu_prop) * phi),
        therm_prop,
    )
    pm.Potential("loglik_prop_mid",  pt.sum(log_p_mid))
    pm.Potential("loglik_prop_beta", pt.sum(logp_beta))



with mock_orderd_beta:
    check_ordered_beta = pm.sample_prior_predictive()


with mock_orderd_beta:
    check_ordered_beta.extend(
        pm.sample(target_accept = 0.95)
    )


summary_vars = [
    "intercept",
    "b_edu", "b_income", "b_edu_income",
    "region_sigma", "log_phi",
    "cutpoint_0", "cutpoint_gap", "cutpoint_1",
]
summ  = az.summary(check_ordered_beta, hdi_prob=0.95, var_names=summary_vars)

summ
