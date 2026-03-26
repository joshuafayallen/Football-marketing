import preliz as pz 
import pymc as pm 

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

n = raw_data['total_plays'].to_numpy()
k = raw_data['n_explosive'].to_numpy()
p_hat = k / n

# under binomial: var(k) = n * p * (1-p)
binomial_var = n * p_hat * (1 - p_hat)
actual_var   = k.var()
mean_binom_var = binomial_var.mean()

print(f"Mean Binomial variance: {mean_binom_var:.4f}")
print(f"Actual variance:        {actual_var:.4f}")
print(f"Overdispersion ratio:   {actual_var / mean_binom_var:.2f}")


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



just_controls = raw_data_pd[predictors]


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
unique_tenure = np.unique(raw_data_pd['tenure_relative'].values)

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

def logit(p):
    return np.log(p / (1 - p))

_epr_raw = raw_data['n_explosive'].to_numpy() / raw_data['total_plays'].to_numpy()
_logit_mean_baseline = float(logit(np.mean(_epr_raw[_epr_raw > 0])))

coords = {
    'seasons': unique_seasons, 
    'predictors': just_controls_sdz.drop('avg_pass_rate',axis = 1).columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index, 
}

lam_prior, _ = pz.maxent(pz.Gamma(), 0.3, 0.9, mass=0.9)


with pm.Model(coords = coords) as mmm_beta_binomial:
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

    n_plays = pm.Data(
        'n_plays', 
        raw_data['total_plays'].to_numpy(), 
        dims = 'obs_id'
    )

    n_explosive = pm.Data(
        'n_explosives',
        raw_data['n_explosive'].to_numpy(),
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

    coach_sigma = pm.HalfNormal('coach_sigma', 0.1)
    coach_baseline = pm.Normal('coach_baseline', 0,1, dims = 'play_callers')
    coach_prior = pm.Normal(
        'coach_prior', 
        mu = _logit_mean_baseline,
        sigma = 0.1,
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
        sigma = 0.1
    )
    mm_alpha_base = pm.Gamma(
        'mm_alpha_base',
        alpha = 3, beta = 6,
        dims='channels'
    )
    
    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = lam_prior.alpha,
        beta = lam_prior.beta, 
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


    mu_logit = pm.math.invlogit(mu)

    pm.Binomial(
        'y_obs',
        p = mu_logit,
        n = n_plays,
        observed = n_explosive, 
        dims = 'obs_id'
    )

with mmm_beta_binomial:
    idata = pm.sample_prior_predictive()

az.plot_ppc(idata, group ='prior', observed = True)

with mmm_beta_binomial:
    idata.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )



with mmm_beta_binomial:
    idata.extend(
        pm.sample_posterior_predictive(idata)
    )

idata.to_netcdf('model/mmm-binomial.nc')





vars = list(idata.posterior.data_vars)
chunks = [vars[i:i + 5] for i in range(0, len(vars) , 5)]


az.plot_trace(
    idata,
    var_names=chunks[0]
)

idata.to_netcdf('model/mmm-beta-binomial.nc')

idata = az.from_netcdf('model/mmm-beta-binomial.nc')

fig, axes = plt.subplots(2, len(coords['channels']), figsize=(16, 8))

channel_names = list(coords['channels'])
for i, ch in enumerate(channel_names):
    # mm_alpha_base
    prior_alpha  = idata.prior['mm_alpha_base'].values.flatten()[ i] \
                   if idata.prior['mm_alpha_base'].ndim > 1 \
                   else idata.prior['mm_alpha_base'].sel(channels=ch).values.flatten()
    post_alpha   = idata.posterior['mm_alpha_base'].sel(channels=ch).values.flatten()

    axes[0, i].hist(prior_alpha, bins=50, alpha=0.5, color='blue', 
                    density=True, label='Prior')
    axes[0, i].hist(post_alpha,  bins=50, alpha=0.5, color='red',  
                    density=True, label='Posterior')
    axes[0, i].set_title(f'mm_alpha_base — {ch}')
    axes[0, i].legend(fontsize=7)

    # mm_lam
    prior_lam  = idata.prior['mm_lam'].sel(channels=ch).values.flatten()
    post_lam   = idata.posterior['mm_lam'].sel(channels=ch).values.flatten()

    axes[1, i].hist(prior_lam, bins=50, alpha=0.5, color='blue',
                    density=True, label='Prior')
    axes[1, i].hist(post_lam,  bins=50, alpha=0.5, color='red',
                    density=True, label='Posterior')
    axes[1, i].set_title(f'mm_lam — {ch}')
    axes[1, i].legend(fontsize=7)

plt.tight_layout()
plt.show()


post = idata.posterior

n_samples = idata.posterior.dims['chain'] * idata.posterior.dims['draw']
coach_mu_s = post['coach_mu'].values          # (chain, draw, obs_id)
control_s  = post['control_contribution'].values
passing_s  = post['passing_contribution'].values 
base_s = coach_mu_s + control_s + passing_s

base_s.shape

# per channel saturated contributions
 # (samples, obs_id, channels)

# build long format dataframe
n_components = len(personnel_cols) + 1  # base + channels

n_chains = idata.posterior.dims['chain']
n_draws = idata.posterior.dims['draw']

n_channels = len(personnel_cols)

x_sat_s = post['x_saturated'].values.reshape(
    n_chains, n_draws, -1, n_channels
)


# cumulative logit surface: (chain, draw, obs_id, n_components)
cumulative_logit = np.zeros((n_chains, n_draws, base_s.shape[2], n_components))
cumulative_logit[:, :, :, 0] = base_s  # base

for ch_i in range(len(personnel_cols)):
    cumulative_logit[:, :, :, ch_i + 1] = (
        cumulative_logit[:, :, :, ch_i] + x_sat_s[:, :, :, ch_i]
    )

base_marginal = expit(base_s)[:, :, :, None]

channel_marginals = np.diff(expit(cumulative_logit), axis=3)

marginals = np.concatenate([base_marginal, channel_marginals], axis=3)

component_names = ['base'] + personnel_cols



obs_coaches = np.array([unique_play_callers[i] for i in coach_idx])
obs_seasons = raw_data_pd['season'].values
obs_weeks   = raw_data_pd['week'].values

rows = {
    'chain':        np.repeat(np.arange(n_chains), n_draws * base_s.shape[2] * n_components),
    'draw':         np.tile(np.repeat(np.arange(n_draws), base_s.shape[2] * n_components), n_chains),
    'obs_id':       np.tile(np.repeat(np.arange(base_s.shape[2]), n_components), n_chains * n_draws),
    'component':    np.tile(component_names, n_chains * n_draws * base_s.shape[2]),
    'contribution': marginals.flatten(),
    'coach':        np.tile(np.repeat(obs_coaches, n_components), n_chains * n_draws),
    'season':       np.tile(np.repeat(obs_seasons, n_components), n_chains * n_draws),
    'week':         np.tile(np.repeat(obs_weeks,   n_components), n_chains * n_draws),
}


rows['coach']  = np.tile(np.repeat(obs_coaches, n_components), n_chains * n_draws)
rows['season'] = np.tile(np.repeat(obs_seasons, n_components), n_chains * n_draws)
rows['week']   = np.tile(np.repeat(obs_weeks,   n_components), n_chains * n_draws)

stacked = pl.DataFrame(rows)

stacked.write_parquet('contributions/stacked_contributions.parquet')


