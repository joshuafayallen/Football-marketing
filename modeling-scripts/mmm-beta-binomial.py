import preliz as pz 
import pymc as pm 
from pymc_marketing.mmm.transformers import( 
    geometric_adstock,
    michaelis_menten,
    root_saturation)
from pymc_marketing.mmm.hsgp import(
    HSGP, 
    SoftPlusHSGP,
    create_m_and_L_recommendations, 
    create_constrained_inverse_gamma_prior,
    create_eta_prior, 
    CovFunc
)
import matplotlib.pyplot as plt 
import polars as pl
import pytensor.tensor as pt
import pytensor
import pytensor.xtensor as ptx
import polars.selectors as cs
import pandas as pd 
import arviz as az
import numpy as np
from patsy import dmatrix
import seaborn as sns
from scipy.special import expit
import xarray as xr 
## for whatever reason my version of pymc marketing doesn't have this 
## so we are just copying it from the source code


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
    raw_data.select(cs.starts_with('personnel')).columns
)

coach_usage_cols = (
    raw_data
    .group_by(['off_play_caller'])
    .agg(
        [
            pl.col(c).mean().alias(c) for c in personnel_cols
        ]
    )
)

long_usage = (
    coach_usage_cols
    .unpivot(
        index = 'off_play_caller', 
        on = personnel_cols,
        variable_name='personnel_group', 
        value_name='mean_usage'
    )
    .filter(pl.col("mean_usage") >= 0.05)
)

keep = long_usage['personnel_group'].unique().to_list()

personnel_scaled = (
    raw_data
    .select(pl.col(keep))
    .with_columns(
        [
            (pl.col(c)/pl.col(c).abs().max()) for c in keep
        ]
    )
)

coach_group_pairs = list(
    zip(long_usage['off_play_caller'].to_list(),
        long_usage['personnel_group'].to_list()
    )
)

n_pairs = len(coach_group_pairs)
pair_coach_idx = np.array(
    pd.Categorical(
        [c for c, _ in coach_group_pairs],
        categories=unique_play_callers  # same categories as coach_idx
    ).codes
)

pair_group_idx = np.array(
    pd.Categorical(
        [g for _, g in coach_group_pairs],
        categories=list(keep)  # same order as coords['channels']
    ).codes
)

coach_match_np = (coach_idx[:, None] == pair_coach_idx[None, :]).astype(np.float32)


unique_tenure = np.unique(raw_data_pd['tenure_relative'].values)

tenure_idx = pd.Categorical(
    raw_data_pd['tenure_relative'], categories=unique_tenure
).codes

tc_max = raw_data_pd['tenure_relative'].max()
tc_mid = tc_max/2

m_seasons, l_seasons  = create_m_and_L_recommendations(
    X = unique_tenure, 
    X_mid=tc_mid,
    ls_lower=2.5, 
    ls_upper=tc_max,
    cov_func=CovFunc.Matern52
)


ls_prior = create_constrained_inverse_gamma_prior(
    lower = 2.5, 
    upper = tc_max, 
    mass = 0.9
)

eta_prior = create_eta_prior(mass = 0.05, upper = 0.2)


def logit(p):
    return np.log(p / (1 - p))

_epr_per_game = (
    raw_data['n_explosive'] / raw_data['total_plays']
).to_numpy()

_epr_per_game = _epr_per_game[(_epr_per_game > 0) & (_epr_per_game < 1)]

_logit_mean_baseline = float(np.mean(logit(_epr_per_game)))
_logit_std_baseline  = float(np.std(logit(_epr_per_game)))



coords = {
    'tenure': unique_tenure, 
    'predictors': just_controls_sdz.drop('avg_pass_rate',axis = 1).columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index, 
    'coach_group_pairs': [f"{c}|{g}" for c,g in coach_group_pairs]

}

coach_obs_counts = (
                    raw_data_pd
                    .groupby('off_play_caller')
                    .size()[unique_play_callers]
                    .values)




with pm.Model(coords = coords) as mmm_hsgp:
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
    tenure_id = pm.Data('tenure_relative',
                        tenure_idx,
                        dims = 'obs_id')

    career_exp_dat = pm.Data(
        'career_exp', 
        raw_data['career_scaled'].to_numpy(), 
        dims = 'obs_id'
    )

    pair_coach_id = pm.Data('pair_coach_id', pair_coach_idx)
    pair_group_id = pm.Data('pair_group_id', pair_group_idx)

    global_baseline = pm.Normal('global_baseline', _logit_mean_baseline, 0.1)

    coach_sigma = pm.HalfNormal('coach_sigma', 0.1)

# Per-coach deviation (non-centered)
    coach_offset_raw= pm.Normal('coach_offset_raw',
                                mu = 0,
                                sigma = 1,
                                dims='play_callers')
    
    coach_offset = pm.Deterministic(
        # add sum to zero constraint
        'coach_offset', 
        coach_offset_raw - coach_offset_raw.mean(), 
        dims = 'play_callers'
    )

    coach_mean = pm.Deterministic(
    'coach_mean',
    global_baseline + (coach_sigma * coach_offset),
    dims='play_callers')


    tenure_hsgp = SoftPlusHSGP(
        eta= eta_prior,
        ls = ls_prior,
        m = 3, 
        L = tc_max * 1.5,
        X = unique_tenure, 
        X_mid = tc_mid,
        cov_func=CovFunc.Matern52,
        dims = 'tenure', 
        centered = False, 
        drop_first=True
    )

    tenure_effect= tenure_hsgp.create_variable('tenure_effect')

    coach_mu = pm.Deterministic(
        'coach_mu', 
        coach_mean[coach_id]
        + tenure_effect[tenure_id],
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
        pm.math.dot(global_controls, controls_prior)
    )
    passing_prior = pm.Normal(
        'passing_prior', 
        mu = 0, 
        sigma = 0.1,
    )

    passing_contribution = pm.Deterministic(
        'passing_contribution', 
        pm.math.dot(passing_dat, passing_prior),
        dims = 'obs_id'
    )

    adstock_alphas = pm.Beta(
        'adstock_alphas', 
        alpha = 1,
        beta = 5, 
        dims = 'channels'
    )

    make_adstocks, _ = pytensor.scan(
        fn = lambda col, alpha: geometric_adstock(
            col, alpha, l_max = 2
        ),
        sequences=[personnel_dat.T, adstock_alphas]
    )
    x_adstock = pm.Deterministic(
        'x_adstock',
        make_adstocks.T,
        dims = ('obs_id', 'channels')
    )

    mm_lam = pm.Gamma(
        'mm_lam',
        alpha = 2, beta = 8,
        dims = 'coach_group_pairs'
    )
    mm_alpha = pm.Gamma(
        'mm_alpha',
        alpha =3,
        beta = 6,
        dims = 'coach_group_pairs'
    )

    x_for_pairs = x_adstock[:, pair_group_id]

    x_saturated_pairs = pm.Deterministic(
        'x_saturated_pairs',
        michaelis_menten(x_for_pairs, mm_alpha[None, :], mm_lam[None, :]),
        dims = ('obs_id', 'coach_group_pairs')
    )

    coach_match = pt.eq(
        coach_id[:, None],
        pair_coach_id[None, :]
    )

    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        (x_saturated_pairs * coach_match).sum(axis =1), 
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
    




with mmm_hsgp:
    idata_hsgp = pm.sample_prior_predictive()


with mmm_hsgp:
    idata_hsgp.extend(
        pm.sample(random_seed=RANDOM_SEED,
        nuts_sampler='nutpie')
    )

idata_hsgp.sample_stats['diverging'].sum().data



with mmm_hsgp:
    pm.sample_posterior_predictive(idata_hsgp, extend_inferencedata=True)
    pm.compute_log_likelihood(idata_hsgp)

az.plot_ppc(idata_hsgp, kind='cumulative')
az.plot_loo_pit(idata_hsgp, y='y_obs', legend=True)

fig, ax = plt.subplots()
az.plot_forest(
    data = [idata_hsgp.prior, idata_hsgp.posterior],
    model_names=['prior', 'posterior'],
    var_names=['mm_lam'],
    combined = True
)
loo = az.loo(idata_hsgp, pointwise=True)
az.plot_khat(loo, show_bins=True)

az.plot_ess(
    idata_hsgp, 
    var_names=['contribution'],
    filter_vars='like', 
    kind = 'evolution'
)


az.plot_ess(
    idata_hsgp,
    var_names=['mm'],
    filter_vars='like',
    kind = 'evolution'
)

az.plot_ess(
    idata_hsgp,
    var_names=['coach', 'global_baseline'],
    filter_vars='like',
    kind = 'evolution'
)
az.plot_ess(
    idata_hsgp,
    var_names=['tenure_effect'],
    #filter_vars='like',
    kind = 'evolution'
)

az.plot_ess(
    idata_hsgp,
    var_names=['adstock_alphas'],
    kind = 'evolution'
)

az.plot_ess(
    idata_hsgp, 
    var_names=['contribution'],
    filter_vars='like', 
    kind = 'evolution'
)


az.plot_ess(
    idata_hsgp,
    var_names=['mm'],
    filter_vars='like',
    kind = 'evolution'
)

az.plot_ess(
    idata_hsgp,
    var_names=['coach', 'global_baseline'],
    filter_vars='like',
    kind = 'evolution'
)
az.plot_ess(
    idata_hsgp,
    var_names=['tenure_effect'],
    #filter_vars='like',
    kind = 'evolution'
)

summary = az.summary(
    idata_hsgp,
    var_names=['adstock_alphas'],
    round_to=0
)[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']]

print(summary)

az.plot_trace(
    idata_hsgp, 
    var_names=['adstock_alphas'],
)


print(az.summary(
    idata_hsgp,
    var_names=['adstock_alphas', 'mm_alpha', 'mm_lam', 
               'global_baseline', 'coach_sigma', 'controls_beta'],
)[['mean', 'sd', 'ess_bulk', 'ess_tail', 'r_hat']])
# everything looks dandy 
idata_hsgp.to_netcdf('model/mmm-binomial.nc')

idata_hsgp = az.from_netcdf('model/mmm-binomial.nc')

coach_group_pairs = (
    pl.from_pandas(
        idata_hsgp.posterior['x_saturated_pairs'].to_dataframe().reset_index()
    )
)


cleanup = (
    coach_group_pairs
    .with_columns(
        pl.col('coach_group_pairs').str.split_exact('|',1)
        .struct.rename_fields(['coach', 'personnel_grouping'])
        .alias('parts')

    )
    .unnest("parts")
    .drop('coach_group_pairs')
)

personnel_contributions = (
    pl.from_pandas(idata_hsgp.posterior['personnel_contribution']
    .to_dataframe()
    .reset_index()
    )
)

coach_mean = (
    pl.from_pandas(
        idata_hsgp.posterior['coach_mean']
        .to_dataframe()
        .reset_index()
    
    )
    .rename({"coach_mean": 'coach_intercept'})
)
coach_mean.columns

overall_mean  = (
    pl.from_pandas(
        idata_hsgp.posterior['mu']
        .to_dataframe()
        .reset_index()
    )
    .rename({'mu': 'overall_mean'})
)

overall_mean.columns 
obs_metadata = (
    raw_data
    .select(['season', 'week', 'off_play_caller', 'tenure_relative', 'total_plays', 'n_explosive'])
    .with_row_index('obs_id')
    .with_columns(
        (pl.col('season') + (pl.col('week')-1)/18).alias('time')
    )
)

add_personnel_contributions = (
    cleanup
    .join(personnel_contributions, on = ['chain', 'draw', 'obs_id'])
    .join(coach_mean, left_on = ['chain', 'draw', 'coach'],
                        right_on = ['chain', 'draw', 'play_callers'])
    .join(overall_mean, on = ['chain', 'draw', 'obs_id'])
    .join(obs_metadata, left_on = ['obs_id', 'coach'],
                        right_on = ['obs_id' ,'off_play_caller'])
    
    .with_columns(
        # make sure i don't get datatype error in R 
        # later 
        pl.col('coach', 'personnel_grouping').cast(pl.String)
    )
)


add_personnel_contributions.write_parquet(
    'contribution',
    partition_by = ['coach', 'personnel_grouping'],
)




idata = az.from_netcdf('model/mmm-binomial.nc')

raw_data_pd['time_decimal'] = (
    raw_data_pd['season'] + (raw_data_pd['week'] - 1) / 18
)


shanny_mask = raw_data_pd['off_play_caller'] == 'Kyle Shanahan'
shanny_obs = np.where(shanny_mask)[0]

time_vals = raw_data_pd.loc[shanny_mask, 'time_decimal'].values

obs_rate =  (raw_data['n_explosive'].to_numpy()[shanny_mask] / 
            raw_data['total_plays'].to_numpy()[shanny_mask])

mu_samples = az.extract(idata, group = 'posterior', var_names=['mu']).values[shanny_obs, :]

p_samples = expit(mu_samples)


df_shanny = raw_data_pd[shanny_mask].copy()

seasons = sorted(df_shanny['tenure_relative'].unique())

n_seasons = len(seasons)

cols = 2

rows = (n_seasons + cols - 1) // cols

fig, axes = plt.subplots(
    rows, cols, figsize=(12, 2.5 * rows), layout="constrained", sharey=True
)

fig, axs = plt.subplots(n_seasons,1 ,sharex=False, sharey= False)    
    





