import preliz as pz 
import pymc as pm 
import pymc.dims as pmd
from pymc_marketing.mmm.transformers import geometric_adstock, michaelis_menten
from pymc_marketing.mmm.hsgp import(
    HSGP, 
    SoftPlusHSGP,
    create_m_and_L_recommendations, 
    create_constrained_inverse_gamma_prior,
    create_eta_prior, 
    CovFunc, 
    plot_curve
)
import matplotlib.pyplot as plt 
import polars as pl
import pytensor.tensor as pt
import pytensor.xtensor as ptx
import polars.selectors as cs
import pandas as pd 
import arviz as az
import numpy as np
from patsy import dmatrix
import seaborn as sns
from scipy.special import expit

seed = 14993111

RANDOM_SEED = np.random.default_rng(seed = seed)

def plot_prior(trace ,param: str):
    fig, axe = plt.subplots()
    return az.plot_dist(trace.prior[param])



keep_these = (
    pl.read_parquet('processed-data/processed-dat.parquet')
    .group_by('off_play_caller')
    .agg(
        pl.len().alias('games_called')
    )
    .filter(
        pl.col('games_called') >= 104 # generally we want at least like two years of data 
    )
    ['off_play_caller'].to_list()
)


raw_data = (
    pl.read_parquet('processed-data/processed-dat.parquet')
    .with_columns(
        pl.col('nflverse_game_id')
        .str.extract(r"_(\d{2})_")
        .str.replace_all('_', '')
        .str.to_integer()
        .alias('week'),
        pl.when(pl.col('surface') == 'grass')
        .then(pl.lit(1))
        .otherwise(0)
        .alias('is_grass'),
        pl.when(
        pl.col('roof').is_in(['closed', 'dome']))
        .then(pl.lit(1))
        .otherwise(0)
        .alias('is_indoors'), 
        pl.col('game_date').str.extract(r"(\d{4})").str.to_integer().alias('year')    
            )
        .with_columns(
            (pl.col('year') - 2017).alias('obs_tenure')
        )
        .filter(
            pl.col('off_play_caller').is_in(keep_these)
        )
        ## we just need to get some relative indicators 
        .with_columns(
            pl.col('play_caller_tenure').min().over('off_play_caller').alias('first_observed')
        )
        .with_columns(
            (pl.col('play_caller_tenure') - pl.col('first_observed'))
            .alias('tenure_relative')
        )
        .with_columns(
            ((pl.col('first_observed') - pl.col('first_observed').mean())/
            pl.col('first_observed').std()).alias('career_scaled')
        )

)   




raw_data_pd = (raw_data.to_pandas()
            .sort_values(['off_play_caller', 'season', 'week']))




unique_seasons = raw_data_pd['play_caller_tenure'].sort_values().unique()

unique_play_callers = raw_data_pd['off_play_caller'].sort_values().unique()

season_idx = pd.Categorical(
    raw_data_pd['play_caller_tenure'], categories=unique_seasons
).codes


coach_idx = pd.Categorical(
    raw_data_pd['off_play_caller'], categories=unique_play_callers).codes


predictors = ['avg_epa', 'avg_defenders_in_box',
            'is_indoors', 'is_grass', 'div_game',
            'wind', 'temp', 'is_home_team', 'avg_diff', 'avg_pass_rate']

plt.subplots()
sns.histplot(raw_data, x = 'avg_diff')

sns.histplot(raw_data, x = 'avg_epa')
sns.histplot(raw_data, x = 'avg_pass_rate')
sns.histplot(raw_data, x = 'avg_defenders_in_box')

fig, ax = plt.subplots(ncols = 2)

sns.histplot(raw_data, x = 'wind', ax = ax[0])

ax[0].set_title('Wind')

sns.histplot(raw_data, x = 'temp',ax = ax[1])
ax[1].set_title('Temp')


just_controls = raw_data_pd[predictors]

nobs = raw_data.height

scaled_y = (
    raw_data
    .with_columns(
    (pl.col('explosive_play_rate') / (pl.col('explosive_play_rate').abs()
    .max()))
    .alias("scaled_explosive_plays")
    ).with_columns(
        (((pl.col('scaled_explosive_plays')) * (nobs-1) + 0.5)/nobs)
        .alias("explosive_play_rate_transformed")
    )

)

print(raw_data['explosive_play_rate'].describe())
print(f"Games with EPR < 0.05: {(raw_data['explosive_play_rate'] < 0.05).sum()}")
print(f"Games with EPR = 0.0:  {(raw_data['explosive_play_rate'] == 0.0).sum()}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
raw_data['explosive_play_rate'].to_pandas().hist(bins=50, ax=axes[0])
axes[0].set_title('Raw explosive play rate')
scaled_y['explosive_play_rate_transformed'].to_pandas().hist(bins=50, ax=axes[1])
axes[1].set_title('Transformed')



get_stds = (
    raw_data
    .group_by('season')
    .agg(
        pl.col('explosive_play_rate').std()
    )

)


numeric_cols = (
    pl.from_pandas(just_controls)
    .select(
            pl.exclude('div_game', 'is_home_team', 'is_indoors', 'is_grass')
            ).columns
)


means = (
    pl.from_pandas(just_controls)
    .select(
        [pl.col(c).mean().alias(c) for c in numeric_cols]
    )

)


stds = (
    pl.from_pandas(just_controls)
    .select(
        [pl.col(c).std().alias(c) for c in numeric_cols]
    )

)



just_controls_pl = pl.from_pandas(just_controls)



just_controls_sdz = (
    just_controls_pl
    .with_columns(
        [((pl.col(c) - means[0,c])/ stds[0, c]).alias(c) for c in numeric_cols]
    )
    .with_columns(
        pl.Series("div_game", just_controls_pl['div_game']),
        pl.Series('is_home_team', just_controls_pl['is_home_team']), 
        pl.Series('is_indoors', just_controls_pl['is_indoors']), 
        pl.Series ('is_grass', just_controls_pl['is_grass'])
    ).to_pandas()

)

personnel_cols = (
    raw_data.select(cs.starts_with('personnel')).to_pandas()
)


usage = personnel_cols.mean(axis = 0)

reliable_cols = usage[usage >= 0.01].index.tolist()

personnel_cols = personnel_cols[reliable_cols]


used_cols = (
    raw_data
    .select(cs.starts_with('personnel'))

)


usage = used_cols.mean()

mask_df = usage.select(
    [(pl.col(c) >= 0.01).alias(c) for c in usage.columns]

)

mask = mask_df.row(0, named = True)

reliable_cols = [name for name, keep in mask.items() if keep]

personnel_cols = used_cols.select(reliable_cols).columns

personnel_scaled = (
    raw_data
    .select(pl.col(personnel_cols))
    .with_columns(
        [
            (pl.col(c) / (pl.col(c).abs().max()))
            for c in personnel_cols
        ]
    ).to_pandas()
    

)

## instead of using a full hsgp we are going to use a penalized one 
## we do also need 

tc_max = raw_data_pd['tenure_relative'].max()
tc_mid = tc_max/2

m_seasons, l_seasons  = create_m_and_L_recommendations(
    X = raw_data_pd['tenure_relative'], 
    X_mid=tc_mid,
    ls_lower=1.0, 
    ls_upper=tc_max,
    cov_func=CovFunc.Matern52
)

ls_prior = create_constrained_inverse_gamma_prior(
    lower = 1.0, 
    upper = tc_max, 
    mass = 0.9
)

eta_prior = create_eta_prior(mass = 0.05, upper = 0.5)



with pm.Model() as prior_mod:
    ls_var = ls_prior.create_variable('ls')
    eta_var = eta_prior.create_variable('eta')
    idata_priors = pm.sample_prior_predictive()



fig, axes = plt.subplots(1, 2, figsize=(12, 4))
draws = idata_priors.prior
ls_samples = draws['ls'].values.flatten()
ax = axes[0]
az.plot_dist(ls_samples, ax=ax, color='steelblue')
ax.axvline(1.0, color='tomato', linestyle='--', linewidth=1.2,
           label='ls_lower = 1 tenure-yr')
ax.axvline(np.median(unique_seasons), color='goldenrod', linestyle=':',
           linewidth=1.2, label=f'X_mid = {tc_mid:.0f}')

# annotate the tail guarantee: P[ls < 1] = 0.9
pct_below = (ls_samples < 1.0).mean()
ax.set_title(
    f'Lengthscale prior  (Weibull + reciprocal)\n'
    f'P[ls < 1] = {pct_below:.2f}  |  median = {np.median(ls_samples):.2f}',
    fontsize=10
)
ax.set_xlabel('Lengthscale (tenure-years)')
ax.legend(fontsize=8)

# --- eta (amplitude) ---
eta_samples = draws['eta'].values.flatten()
ax = axes[1]
az.plot_dist(eta_samples, ax=ax, color='mediumseagreen')
ax.axvline(1.0, color='tomato', linestyle='--', linewidth=1.2,
           label='upper = 1.0')

pct_above = (eta_samples > 1.0).mean()
ax.set_title(
    f'Lengthscale prior  (InverseGamma, constrained)\n'
    f'P[ls < 1] = {pct_below:.2f}  |  median = {np.median(ls_samples):.2f}',
    fontsize=10
)

ax.set_xlabel('Eta (GP amplitude, logit scale)')
ax.legend(fontsize=8)

fig.suptitle(
    f'HSGP priors  |  m={m_seasons}, L={l_seasons:.2f}, X_mid={tc_mid:.1f}',
    fontsize=11
)
plt.tight_layout()
plt.show()


def logit(p):
    return np.log(p / (1 - p))

_obs_epr = scaled_y['explosive_play_rate_transformed'].to_numpy()
_logit_mean_baseline = float(logit(np.mean(_obs_epr))) 

coords = {
    'seasons': unique_seasons, 
    'predictors': just_controls_sdz.drop('avg_pass_rate',axis = 1).columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index, 
}

with pm.Model(coords = coords) as mmm:

    global_controls = pm.Data(
        'control_data',
        just_controls_sdz.drop('avg_pass_rate', axis = 1),
        dims = ('obs_id', 'predictors')
    )

    passing_dat = pm.Data(
        'passing_data', 
        just_controls_sdz['avg_pass_rate'], 
        dims = 'obs_id'
    )

    personnel_dat = pm.Data(
        'personnel_data', 
        personnel_scaled.to_numpy(),
        dims = ('obs_id', 'channels')
    )

    obs_exp_plays = pm.Data(
        'obs_exp_plays',
        scaled_y['explosive_play_rate_transformed'].to_numpy(),
        dims = 'obs_id'
    )

    coach_id = pm.Data('coach_id', coach_idx, dims = 'obs_id')
    tenure_dat = pm.Data('tenure_relative',
                        raw_data['tenure_relative'].to_numpy(),
                        dims = 'obs_id')
    
    career_exp_dat = pm.Data(
        'career_exp', 
        raw_data['career_scaled'].to_numpy(), 
        dims = 'obs_id'
    )

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5),
    coach_baseline = pm.Normal('coach_baseline', 0,1, dims = 'play_callers')
    coach_prior = pm.Normal(
        'coach_prior', 
        mu = _logit_mean_baseline,
        sigma = 0.5,
    )

    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_prior + (coach_sigma * coach_baseline), 
        dims = 'play_callers'

    )
    coach_mu = pm.Deterministic(
        'coach_mu', 
        coach_mean[coach_id],
        dims = 'obs_id'
    )

    controls_prior = pm.Normal(
        'controls_beta', 
        mu = 0,
        sigma = 0.05,
        dims = 'predictors'
    )

    control_contribution = pm.Deterministic(
        'control_contribution',
        pm.math.dot(global_controls, controls_prior),
        dims = 'obs_id'
    )
    

    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)
    passing_contribution = pm.Deterministic(
        'passing_contribution', 
        pm.math.dot(passing_dat, passing_prior),
        dims = 'obs_id'
    )

    adstock_alphas = pm.Beta(
        'adstock_alphas',
        alpha = 1, 
        beta = 12,
        dims = 'channels'
    )

    adstock_list = []

    for i in range(len(coords['channels'])):
        adstock_list.append(
            geometric_adstock(personnel_dat[:,i], adstock_alphas[i], l_max = 2)
        )
    
    x_adstock = pm.Deterministic(
        'x_adstock', 
        pt.stack(adstock_list, axis = 1), 
        dims = ('obs_id', 'channels')
    )

    exp_effect = pm.Normal(
        'exp_effect', 
        mu = 0, 
        sigma = 0.5
    )
    mm_alpha_base = pm.Gamma(
        'mm_alpha_base',
        alpha = 3, beta = 6,
        dims='channels'
    )
    
    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = 3,
        beta = 15, 
        dims = 'channels'
    )

    mm_alpha = pm.Deterministic(
        'mm_alpha', 
        mm_alpha_base[None, :] * pm.math.exp(
            exp_effect * career_exp_dat[:,None]
        ),
        dims = ('obs_id', 'channels')
    )

    saturated_list = []
    for i in range(len(coords['channels'])):
        saturated_list.append(
            michaelis_menten(x_adstock[:,i], mm_alpha[:,i], mm_lam[i])
        )
    
    x_saturated_base = pt.stack(saturated_list, axis = 1)

    hsgp_tvp = SoftPlusHSGP(
        eta = eta_prior,
        ls = ls_prior, 
        m = m_seasons,
        L = l_seasons,
        X = tenure_dat,
        X_mid=tc_mid,
        cov_func=CovFunc.Matern52,
        dims = ('obs_id', 'channels'),
        centered=False,
        drop_first=False
    )
    tenure_mult = hsgp_tvp.create_variable(
        'tenure_multiplier'
    )

    x_saturated = pm.Deterministic(
        'x_saturated', 
        x_saturated_base * tenure_mult,
        dims = ('obs_id', 'channels')
    )

    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        x_saturated.sum(axis =1),
        dims = 'obs_id'
    )

    mu = pm.Deterministic(
        'mu',
        coach_mu + 
        personnel_contribution + 
        control_contribution + 
        passing_contribution, 
        dims = 'obs_id'
    )

    precision = pm.Gamma('precision', alpha = 10, beta = 0.5)
    # precisions = pm.Exponential('precision', 1/25)

    mu_logit = pm.math.invlogit(mu)

    pm.Beta(
        'y_obs', 
        alpha = mu_logit * precision,
        beta = (1-mu_logit) * precision,
        observed=obs_exp_plays,
        dims = 'obs_id'
    )


with mmm:
    idata = pm.sample_prior_predictive()

az.plot_ppc(idata, group = 'prior', observed = True)

with mmm:
    idata.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )


with mmm:
    idata.extend(
        pm.sample_posterior_predictive(
            idata
        )
    )
    pm.compute_log_likelihood(idata)

az.plot_ppc(idata)




unique_tenure = raw_data_pd['tenure_relative'].sort_values().unique()
_tenure_idx = pd.Categorical(
    raw_data_pd['tenure_relative'], categories=unique_tenure
).codes

coords['tenure_grid'] = unique_tenure

with pm.Model(coords = coords) as mmm_non_mult:
    global_controls = pm.Data(
        'control_data',
        just_controls_sdz.drop('avg_pass_rate', axis = 1),
        dims = ('obs_id', 'predictors')
    )

    passing_dat = pm.Data(
        'passing_data', 
        just_controls_sdz['avg_pass_rate'], 
        dims = 'obs_id'
    )

    personnel_dat = pm.Data(
        'personnel_data', 
        personnel_scaled.to_numpy(),
        dims = ('obs_id', 'channels')
    )

    obs_exp_plays = pm.Data(
        'obs_exp_plays',
        scaled_y['explosive_play_rate_transformed'].to_numpy(),
        dims = 'obs_id'
    )

    coach_id = pm.Data('coach_id', coach_idx, dims = 'obs_id')
    
    tenure_id = pm.Data('tenure_relative',
                        _tenure_idx,
                        dims = 'obs_id')
    
    career_exp_dat = pm.Data(
        'career_exp', 
        raw_data['career_scaled'].to_numpy(), 
        dims = 'obs_id'
    )

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5)
    coach_baseline = pm.Normal('coach_baseline', 0,1, dims = 'play_callers')
    coach_prior = pm.Normal(
        'coach_prior', 
        mu = _logit_mean_baseline,
        sigma = 0.5,
    )
    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_baseline + coach_sigma * coach_prior, 
        dims = 'play_callers'

    )
    hsgp_tvp = SoftPlusHSGP(
        eta = eta_prior,
        ls = ls_prior, 
        m = m_seasons,
        L = l_seasons,
        X = unique_tenure,
        X_mid=tc_mid,
        cov_func=CovFunc.Matern52,
        dims = 'tenure_grid',
        centered=False,
        drop_first=True
    )
    tenure_effect = hsgp_tvp.create_variable('tenure_effect')

    coach_mu = pm.Deterministic(
        'coach_mu', 
        coach_mean[coach_id] + tenure_effect[tenure_id], 
        dims = 'obs_id'
    )

    controls_prior = pm.Normal(
        'controls_beta', 
        mu = 0,
        sigma = 0.05,
        dims = 'predictors'
    )

    control_contribution = pm.Deterministic(
        'control_contribution',
        pm.math.dot(global_controls, controls_prior),
        dims = 'obs_id'
    )

    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)

    adstock_alphas = pm.Beta(
        'adstock_alphas',
        alpha = 1, 
        beta = 12,
        dims = 'channels'
    )

    adstock_list = []

    for i in range(len(coords['channels'])):
        adstock_list.append(
            geometric_adstock(personnel_dat[:,i], adstock_alphas[i], l_max = 2)
        )
    
    x_adstock = pm.Deterministic(
        'x_adstock', 
        pt.stack(adstock_list, axis = 1), 
        dims = ('obs_id', 'channels')
    )
    mm_alpha = pm.Gamma(
        'mm_alpha',
        alpha = 3, beta = 6,
        dims='channels'
    )
    
    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = 3,
        beta = 15, 
        dims = 'channels'
    )

    saturated_list = []
    for i in range(len(coords['channels'])):
        saturated_list.append(
            michaelis_menten(x_adstock[:,i], mm_alpha[i], mm_lam[i])
        )
    
    x_saturated = pm.Deterministic(
        'x_saturated',
        pt.stack(saturated_list, axis = 1),
        dims=('obs_id', 'channels')
    )

    personnel_contribution= pm.Deterministic(
        'personnel_contribution',
        x_saturated.sum(axis=1),
        dims = 'obs_id'
    )
    mu = pm.Deterministic(
        'mu',
        coach_mu +
        personnel_contribution +
        control_contribution +
        passing_prior * passing_dat,
        dims='obs_id'
    )

    precision = pm.Gamma('precision', alpha=10, beta=0.5, shape=())
    mu_logit  = pm.math.invlogit(mu)
    pm.Beta(
        'y_obs',
        alpha=mu_logit * precision,
        beta=(1 - mu_logit) * precision,
        observed=obs_exp_plays,
        dims='obs_id'
    )



with mmm_non_mult:
    idata2 = pm.sample_prior_predictive()


az.plot_ppc(idata2, group = 'prior', observed = True)

with mmm_non_mult:
    idata2.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )


    
with mmm_non_mult:
    idata2.extend(
        pm.sample_posterior_predictive(idata2)
    )
    pm.compute_log_likelihood(idata2)

az.plot_ppc(idata2)


len(unique_tenure)

n_knots = 3

knots = np.quantile(unique_tenure, np.linspace(0,1, n_knots))
spline_obs = dmatrix(
    "bs(tenure, knots = knots, degree = 3, include_intercept = True) - 1",
    {'tenure': raw_data['tenure_relative'].to_numpy(), 'knots':knots[1:-1]}
)
spline_grid = dmatrix(
    "bs(tenure, knots=knots, degree=3, include_intercept=True) - 1",
    {'tenure': unique_tenure, 'knots': knots[1:-1]}
)


basis_set_obs  = np.array(spline_obs)   # (obs_id, n_basis)
basis_set_grid = np.array(spline_grid)  # (n_unique_tenure, n_basis)

coords['spline_basis'] = [f"s{i}" for i in range(basis_set_obs.shape[1])]

with pm.Model(coords = coords) as mmm_spline:
    global_controls = pm.Data(
        'control_data', just_controls_sdz.drop('avg_pass_rate', axis=1),
        dims=('obs_id', 'predictors')
    )
    passing_dat   = pm.Data('passing_data',
                            just_controls_sdz['avg_pass_rate'], dims='obs_id')
    personnel_dat = pm.Data(
        'personnel_data', personnel_scaled, dims=('obs_id', 'channels')
    )
    obs_exp_plays = pm.Data('obs_exp_plays',
                        scaled_y['explosive_play_rate_transformed'].to_numpy(), 
                        dims='obs_id')

    # Spline basis evaluated at each observation's tenure_relative position
    spline_basis_dat = pm.Data(
        'spline_basis_obs', basis_set_obs,
        dims=('obs_id', 'spline_basis')
    )


    coach_id = pm.Data('coach_id', coach_idx, dims = 'obs_id')
    tenure_dat = pm.Data('tenure_relative',
                        raw_data['tenure_relative'].to_numpy(),
                        dims = 'obs_id')
    
    career_exp_dat = pm.Data(
        'career_exp', 
        raw_data['career_scaled'].to_numpy(), 
        dims = 'obs_id'
    )

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5)
    coach_baseline = pm.Normal('coach_baseline', 0,1, dims = 'play_callers')
    coach_prior = pm.Normal(
        'coach_prior', 
        mu = _logit_mean_baseline,
        sigma = 0.5,
    )

    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_baseline + coach_sigma * coach_prior, 
        dims = 'play_callers'

    )
    coach_mu = pm.Deterministic(
        'coach_mu', 
        coach_mean[coach_id],
        dims = 'obs_id'
    )
    controls_prior = pm.Normal(
        'controls_beta', 
        mu = 0,
        sigma = 0.05,
        dims = 'predictors'
    )

    control_contribution = pm.Deterministic(
        'control_contribution',
        pm.math.dot(global_controls, controls_prior),
        dims = 'obs_id'
    )

    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)

    adstock_alphas = pm.Beta(
        'adstock_alphas',
        alpha = 1, 
        beta = 12,
        dims = 'channels'
    )

    adstock_list = []

    for i in range(len(coords['channels'])):
        adstock_list.append(
            geometric_adstock(personnel_dat[:,i], adstock_alphas[i], l_max = 2)
        )
    
    x_adstock = pm.Deterministic(
        'x_adstock', 
        pt.stack(adstock_list, axis = 1), 
        dims = ('obs_id', 'channels')
    )
    exp_effect = pm.Normal(
        'exp_effect', 
        mu = 0, 
        sigma = 0.5
    )
    mm_alpha_base = pm.Gamma(
        'mm_alpha_base',
        alpha = 3, beta = 6,
        dims='channels'
    )
    
    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = 3,
        beta = 15, 
        dims = 'channels'
    )

    mm_alpha = pm.Deterministic(
        'mm_alpha', 
        mm_alpha_base[None, :] * pm.math.exp(
            exp_effect * career_exp_dat[:,None]
        ),
        dims = ('obs_id', 'channels')
    )

    saturated_list = []
    for i in range(len(coords['channels'])):
        saturated_list.append(
            michaelis_menten(x_adstock[:,i], mm_alpha[:,i], mm_lam[i])
        )
    
    x_saturated_base = pt.stack(saturated_list, axis = 1)
    
    spline_sigma = pm.Exponential('spline_sigma', 2)
    spline_raw = pm.Normal('spline_raw',
                            mu = 0,
                            sigma = 1,
                            dims = ('spline_basis', 'channels'))


    spline_beta = pm.Deterministic(
        'spline_beta', 
        spline_sigma * spline_raw, 
        dims = ('spline_basis', 'channels')
    )


    spline_surface = pm.math.dot(spline_basis_dat, spline_beta)
    tenure_factor_raw = pm.math.log1pexp(spline_surface)
    tenure_factor_mean = tenure_factor_raw.mean(axis = 0)

    tenure_mult = pm.Deterministic(
        'tenure_mult',
        tenure_factor_raw/tenure_factor_mean[None, :], 
        dims = ('obs_id', 'channels')
    )

    x_saturated = pm.Deterministic(
        'x_saturated', 
        x_saturated_base * tenure_mult,
        dims = ('obs_id', 'channels')
    )


    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        x_saturated.sum(axis =1),
        dims = 'obs_id'
    )

    mu = pm.Deterministic(
        'mu',
        coach_mu + 
        personnel_contribution + 
        control_contribution + 
        pm.math.dot(passing_prior, passing_dat), 
        dims = 'obs_id'
    )

    precision = pm.Gamma('precision', alpha = 10, beta = 0.5)
    # precisions = pm.Exponential('precision', 1/25)

    mu_logit = pm.math.invlogit(mu)

    pm.Beta(
        'y_obs', 
        alpha = mu_logit * precision,
        beta = (1-mu_logit) * precision,
        observed=obs_exp_plays,
        dims = 'obs_id'
    )


with mmm_spline: 
    idata3 = pm.sample_prior_predictive()


az.plot_ppc(idata3, group = 'prior', observed = True)

with mmm_spline:
    idata3.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )


with mmm_spline:
    idata3.extend(
        pm.sample_posterior_predictive(idata3)
    )
    pm.compute_log_likelihood(idata3)

az.plot_ppc(idata3)


with pm.Model(coords = coords) as mmm_spline_coach_mu:
    global_controls = pm.Data(
        'control_data', just_controls_sdz.drop('avg_pass_rate', axis=1),
        dims=('obs_id', 'predictors')
    )
    passing_dat   = pm.Data('passing_data',
                            just_controls_sdz['avg_pass_rate'], dims='obs_id')
    personnel_dat = pm.Data(
        'personnel_data', personnel_scaled, dims=('obs_id', 'channels')
    )
    obs_exp_plays = pm.Data('obs_exp_plays',
                        scaled_y['explosive_play_rate_transformed'].to_numpy(), 
                        dims='obs_id')

    # Spline basis evaluated at each observation's tenure_relative position
    spline_basis_dat = pm.Data(
        'spline_basis_obs', basis_set_obs,
        dims=('obs_id', 'spline_basis')
    )


    coach_id = pm.Data('coach_id', coach_idx, dims = 'obs_id')
    tenure_dat = pm.Data('tenure_relative',
                        raw_data['tenure_relative'].to_numpy(),
                        dims = 'obs_id')
    
    career_exp_dat = pm.Data(
        'career_exp', 
        raw_data['career_scaled'].to_numpy(), 
        dims = 'obs_id'
    )

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5)
    coach_baseline = pm.Normal('coach_baseline', 0,1, dims = 'play_callers')
    coach_prior = pm.Normal(
        'coach_prior', 
        mu = _logit_mean_baseline,
        sigma = 0.5,
    )

    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_prior + (coach_sigma * coach_baseline), 
        dims = 'play_callers'

    )
    spline_sigma = pm.Exponential('spline_sigma', 2)
    spline_raw   = pm.Normal('spline_raw', 0, 1,
                            shape=basis_set_obs.shape[1])
    spline_beta  = pm.Deterministic(
        'spline_beta',
        spline_sigma * spline_raw,
        dims='spline_basis'
    )
    tenure_raw  = pm.math.dot(spline_basis_dat, spline_beta)  # (obs_id,)
    tenure_pos  = pm.math.log1pexp(tenure_raw)                # positive
    tenure_effect = pm.Deterministic(
        'tenure_effect',
        tenure_pos / tenure_pos.mean(),                       # centered at 1
        dims='obs_id'
    )


    coach_mu = pm.Deterministic(
        'coach_mu', 
        coach_mean[coach_id]
        + tenure_effect,
        dims = 'obs_id'
    )
    controls_prior = pm.Normal(
        'controls_beta', 
        mu = 0,
        sigma = 0.05,
        dims = 'predictors'
    )

    control_contribution = pm.Deterministic(
        'control_contribution',
        pm.math.dot(global_controls, controls_prior),
        dims = 'obs_id'
    )

    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)

    adstock_alphas = pm.Beta(
        'adstock_alphas',
        alpha = 1, 
        beta = 12,
        dims = 'channels'
    )

    adstock_list = []

    for i in range(len(coords['channels'])):
        adstock_list.append(
            geometric_adstock(personnel_dat[:,i], adstock_alphas[i], l_max = 2)
        )
    
    x_adstock = pm.Deterministic(
        'x_adstock', 
        pt.stack(adstock_list, axis = 1), 
        dims = ('obs_id', 'channels')
    )
    exp_effect = pm.Normal(
        'exp_effect', 
        mu = 0, 
        sigma = 0.5
    )
    mm_alpha = pm.Gamma(
        'mm_alpha',
        alpha = 3, beta = 6,
        dims='channels'
    )
    
    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = 3,
        beta = 15, 
        dims = 'channels'
    )


    saturated_list = []
    for i in range(len(coords['channels'])):
        saturated_list.append(
            michaelis_menten(x_adstock[:,i], mm_alpha[i], mm_lam[i])
        )
    

    x_saturated = pm.Deterministic(
        'x_saturated', 
        pt.stack(saturated_list, axis = 1),
        dims = ('obs_id', 'channels')
    )


    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        x_saturated.sum(axis =1),
        dims = 'obs_id'
    )

    mu = pm.Deterministic(
        'mu',
        coach_mu + 
        personnel_contribution + 
        control_contribution + 
        pm.math.dot(passing_prior, passing_dat), 
        dims = 'obs_id'
    )

    precision = pm.Gamma('precision', alpha = 10, beta = 0.5)
    # precisions = pm.Exponential('precision', 1/25)

    mu_logit = pm.math.invlogit(mu)

    pm.Beta(
        'y_obs', 
        alpha = mu_logit * precision,
        beta = (1-mu_logit) * precision,
        observed=obs_exp_plays,
        dims = 'obs_id'
    )


with mmm_spline_coach_mu: 
    idata4 = pm.sample_prior_predictive()

az.plot_ppc(idata4, group = 'prior', observed = True)


with mmm_spline_coach_mu:
    idata4.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )



with mmm_spline_coach_mu:
    idata4.extend(
        pm.sample_posterior_predictive(idata4)
    )
    pm.compute_log_likelihood(idata4)




az.plot_ppc(idata4)


mod_names = ['HSGP in Saturation', 'HSGP in Coach Mean', 'Spline in Saturation', 'Spline in Coach Mean']

mods_dict = dict(zip(mod_names, [idata, idata2, idata3, idata4]))


az.compare(mods_dict)

sort_order = raw_data_pd.sort_values(['season', 'week']).index.values

fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for ax, idata, label in zip(
    axs,
    [idata, idata3],
    ['HSGP', 'Spline']
):
    mu_samples = az.extract(idata, group='posterior', var_names='mu')
    mu_prob    = expit(mu_samples.values[sort_order, :])  # (n_obs, samples)

    pm.gp.util.plot_gp_dist(
        ax=ax,
        samples=mu_prob.T,                               # (samples, n_obs)
        x=np.arange(len(sort_order)),
        plot_samples=False,
    )
    ax.plot(
        np.arange(len(sort_order)),
        _obs_epr[sort_order],
        'ok', alpha=0.3, markersize=2, label='Observed EPR'
    )
    ax.set_title(f'Posterior predictive mean — {label}')
    ax.set_ylabel('P(explosive play)')
    ax.legend(fontsize=8)

axs[1].set_xlabel('Game (sorted by season, week)')
plt.tight_layout()
plt.show()