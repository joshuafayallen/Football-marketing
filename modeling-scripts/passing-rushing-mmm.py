from hmac import compare_digest
import preliz as pz 
import pymc as pm 
from pymc_marketing.mmm.transformers import geometric_adstock, michaelis_menten
import matplotlib.pyplot as plt 
import polars as pl
import pytensor.tensor as pt
import polars.selectors as cs
import pandas as pd 
import arviz as az
import numpy as np
import patsy as pa
import seaborn as sns

seed = 14993111

RANDOM_SEED = np.random.default_rng(seed = seed)

def plot_prior(trace ,param: str):
    fig, axe = plt.subplots()
    return az.plot_dist(trace.prior[param])


keep_these = (
    pl.read_parquet('processed-data/explosives-separated.parquet')
    .unique(['nflverse_game_id', 'off_play_caller'])
    .group_by('off_play_caller')
    .agg(
        pl.len().alias('games_called')
    )
    .filter(
        # generally we want at least like two years of data 
        pl.col('games_called') >= 100
    )
    ['off_play_caller'].to_list()
)

predictors = ['avg_epa', 'avg_defenders_in_box',
            'is_indoors', 'is_grass', 'div_game',
            'wind', 'temp', 'is_home_team', 'avg_diff', 'avg_cpoe']

raw_df = (
    pl.read_parquet('processed-data/explosives-separated.parquet')
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
        .alias('is_indoors'))
        .filter(
            pl.col('off_play_caller').is_in(keep_these)
        )
)



unique_seasons = (
    raw_df
    .select(
        pl.col('play_caller_tenure')
    )
    .unique('play_caller_tenure')
    .to_series()
    .sort()
    .to_list()
)

unique_play_callers = (
    raw_df
    .select(
        pl.col("off_play_caller")
    )
    .unique('off_play_caller')
    .to_series()
    .sort()
    .to_list()
)

unique_play_types = (
    raw_df
    .select(
        pl.col('play_type').unique()
    )
    .to_series()
    .sort()
    .to_list()
)


make_encoded_vars = (
    raw_df
    .with_columns(
        pl.col("off_play_caller")
        .cast(pl.Enum(unique_play_callers))
        .to_physical()
        .alias('coach_idx'), 
        pl.col('play_caller_tenure')
        .cast(pl.Enum([str(int(s)) for s in unique_seasons]))  
        .to_physical()
        .alias('season_idx'), 
        pl.col('play_type')
        .cast(pl.Enum(unique_play_types))
        .to_physical()
        .alias('play_type_idx')
    )
)





rush = (
    make_encoded_vars
    .filter(pl.col('play_type') == 'run')
    .rename({
        'explosive_play_rate': 'explosive_play_rate_rush',
        **{c: f"{c}_rush" for c in raw_df.columns if c.startswith('personnel')}
    })
    .drop('play_type')
)

pass_ = (
    make_encoded_vars
    .filter(pl.col('play_type') == 'pass')
    .rename({
        'explosive_play_rate': 'explosive_play_rate_pass',
        **{c: f"{c}_pass" for c in raw_df.columns if c.startswith('personnel')}
    })
    .drop('play_type')
)




wide_df = (
    rush.join(
    pass_.select(['nflverse_game_id', 'off_play_caller'] + 
                [c for c in pass_.columns if c.endswith('_pass')]),
    on=['nflverse_game_id', 'off_play_caller'],
    how='left')
    .with_columns(
        pl.when(
            (pl.col("spread_line") > 0 ) & 
            (pl.col('is_home_team') == 1)
        )
        .then(pl.lit(1))
        .when(
            (pl.col('spread_line') < 0) & 
            (pl.col('is_home_team') == 0)
            )
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias('is_favored')
        
    )
    
)


predictors = ['avg_epa', 'avg_defenders_in_box',
            'is_indoors', 'is_grass', 'div_game',
            'wind', 'temp', 'is_home_team', 'avg_diff', 'is_favored', 'avg_cpoe']


just_controls = (
    wide_df.select(
        pl.col(predictors)
    )
)

numeric_cols = (
    just_controls
    .select(pl.exclude('div_game',
                        'is_home_team',
                        'is_indoors',
                        'is_grass', 
                        'is_favored')).columns
)

wide_df.columns

means = (
    just_controls
    .select(
        [pl.col(c).mean().alias(c) for c in numeric_cols]
    )
)

sds = (
    just_controls
    .select(
        [pl.col(c).std().alias(c) for c in numeric_cols]
    )
)


just_controls_sdz = (
    just_controls
    .with_columns(
        [
        ((pl.col(c) - means[0,c])/ sds[0, c]).alias(c) for c in numeric_cols
        ]
    )
    .with_columns(
        pl.Series("div_game", just_controls['div_game']),
        pl.Series('is_home_team', just_controls['is_home_team']), 
        pl.Series('is_indoors', just_controls['is_indoors']), 
        pl.Series ('is_grass', just_controls['is_grass']),
        pl.Series('is_favored', just_controls['is_favored'] )
    )
)


scaled_y = (
    wide_df
    .select(
        pl.col('explosive_play_rate_rush', 'explosive_play_rate_pass')
    )
    .with_columns(
        (pl.col('explosive_play_rate_rush') +
        pl.col('explosive_play_rate_pass')).alias('overall_explosive_play_rate')
    )
    .with_columns(
        (pl.col('explosive_play_rate_rush', 'explosive_play_rate_pass', 
        'overall_explosive_play_rate')
        / pl.col('explosive_play_rate_rush',
                'explosive_play_rate_pass',
                'overall_explosive_play_rate').abs().max()
        ).name.prefix('scaled_'),
        
    )
    .with_columns(
        pl.col('scaled_explosive_play_rate_rush', 
        'scaled_explosive_play_rate_pass', 
        'scaled_overall_explosive_play_rate')
        .count()
        .name.prefix('nobs_')
    )
    .with_columns(
        ((pl.col('scaled_explosive_play_rate_rush',
                'scaled_explosive_play_rate_pass',
                'scaled_overall_explosive_play_rate')
          * (pl.col('nobs_scaled_explosive_play_rate_rush', 
        'nobs_scaled_explosive_play_rate_pass',
        'nobs_scaled_overall_explosive_play_rate') - 1)
        + 0.5)
        / pl.col('nobs_scaled_explosive_play_rate_rush', 
        'nobs_scaled_explosive_play_rate_pass',
        'nobs_scaled_overall_explosive_play_rate')
        ).name.suffix('_transformed')
    ).rename(
        {
            'scaled_explosive_play_rate_rush_transformed': 'explosive_rush', 
            'scaled_explosive_play_rate_pass_transformed': 'explosive_pass',
            'scaled_overall_explosive_play_rate_transformed': 'explosive_play_rate'
        }
    )
)



personnel_cols = (
    wide_df
    .select(
        cs.starts_with('personnel')
    )
)

usage = personnel_cols.mean()

reliable = (
    usage.transpose(include_header=True)
    .filter(pl.col('column_0') > 0.01)
    .get_column('column')
    .to_list()
)
reliable

personnel_scaled = (
    wide_df
    .select(pl.col(reliable))
    .with_columns(
        [
            pl.col(c) / (pl.col(c).abs().max())
            for c in reliable 
        ]
    ).drop('personnel_10_pass') # dropping this to be consistent between rush and pass
)


between_season_gp, _ = pz.maxent(pz.InverseGamma(), 2, 5)


between_season_m, between_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [
            make_encoded_vars.select(pl.col("play_caller_tenure").min()).item(),
            make_encoded_vars.select(pl.col("play_caller_tenure").max()).item()],
    lengthscale_range=[2,5],
    cov_func='matern52'
)

rush_personnel_cols = sorted([c for c in personnel_scaled.columns if c.endswith('_rush')])

channel_names = [c.replace('_rush', '') for c in rush_personnel_cols]

unique_seasons_num = (
    wide_df.select(
        pl.col('play_caller_tenure').unique()
    )
    .to_series()
    .sort()
    .to_numpy()
)

fig, axe = plt.subplots()
sns.histplot(wide_df, x = 'avg_cpoe')

fig, axe = plt.subplots()
sns.histplot(wide_df, x = 'avg_pass_rate')


weight_data = (
    wide_df
    .select(
        pl.col('total_line', 'spread_line', 'is_home_team',
                'avg_pass_rate', 'avg_cpoe', 'is_favored', 'avg_defenders_in_box')
    )
    .with_columns(
        pl.when(
            pl.col("is_home_team") == 1
        )
        .then(
            pl.col('spread_line')
        )
        .otherwise((pl.col('spread_line') * - 1))
    )
    .with_columns(
        ((pl.col('total_line', 
                'spread_line',
                'avg_pass_rate',
                'avg_cpoe',
                'avg_defenders_in_box') -
        pl.col('total_line',
                'spread_line',
                'avg_pass_rate',
                'avg_cpoe',
                'avg_defenders_in_box').mean()) / 
        pl.col("total_line",
                'spread_line',
                'avg_pass_rate',
                'avg_cpoe',
                'avg_defenders_in_box').std()).
        name.suffix('_sdz')
    ).drop('is_home_team')
)

spread_basis = pa.bs(
    weight_data['spread_line'].to_numpy(),
    df = 6, 
    include_intercept = False
)

total_basis = pa.bs(
    weight_data['total_line'].to_numpy(),
    df = 4,
    include_intercept = False
)



def logit(p):
    return np.log(p / (1 - p))

base_exp = (
    scaled_y['overall_explosive_play_rate'].mean()
)


obs_pass_prior = (
    logit(
    scaled_y['explosive_play_rate_pass'].mean()
    )
)


obs_rush_prior = (
    logit(
    scaled_y['explosive_play_rate_rush'].mean()
    )
)


spread_basis = pa.bs(
    weight_data['spread_line'].to_numpy(),
    df = 6, 
    include_intercept = False
)

total_basis = pa.bs(
    weight_data['total_line'].to_numpy(),
    df = 4,
    include_intercept = False
)
coords_pl = {
    'play_type': ['pass', 'rush'], 
    'seasons': unique_seasons_num, 
    'channels': channel_names, 
    'predictors': just_controls_sdz.columns, 
    'play_callers': make_encoded_vars['off_play_caller'].unique().sort().to_list(), 
    'obs_id': np.arange(wide_df.height),
    'weight_predictors': weight_data.select(pl.col(
    'avg_pass_rate_sdz', 'avg_cpoe_sdz', 
    'avg_defenders_in_box_sdz')).columns,
    'spread_basis': [f"spread_{i}" for i in range(spread_basis.shape[1])], 
    'total_basis': [f"total_{i}" for i in range(total_basis.shape[1])]
}

with pm.Model(coords = coords_pl) as mmm_pass_rush:

    weights_data = pm.Data(
        'weights_data', 
        weight_data.select(pl.col(
        'avg_pass_rate_sdz', 'avg_cpoe_sdz',
        'avg_defenders_in_box_sdz')),
        dims = ('obs_id', 'weight_predictors')
    )

    control_data = pm.Data(
        'control_data', 
        just_controls_sdz.to_numpy(), dims = ('obs_id', 'predictors')
    )
    personnel_rush_data = pm.Data(
        'personnel_rush',
        personnel_scaled.select(cs.ends_with('rush')).to_numpy(),
        dims = ('obs_id', 'channels')
    )

    personnel_pass_data = pm.Data(
        'personnel_pass', 
        personnel_scaled.select(cs.ends_with('pass')).to_numpy(),
        dims = ('obs_id', 'channels')
    )

    seasons_idx = pm.Data(
        'season_id', 
        wide_df['season_idx'].to_numpy().squeeze()
    )

    coach_idx = pm.Data(
        'coach_id',
        wide_df['coach_idx'].to_numpy().squeeze()
    )

    x_seasons = pm.Data(
        'x_seasons', unique_seasons_num, dims = 'seasons'
    )[:, None]

    rush_obs = pm.Data(
        'rush_obs', 
        scaled_y["explosive_rush"].to_numpy(),
        dims = 'obs_id'

    )

    pass_obs = pm.Data(
        'pass_obs',
        scaled_y['explosive_pass'].to_numpy(),
        dims = 'obs_id'
    )

    overall_exp = pm.Data(
        'overall_obs',
        scaled_y['explosive_play_rate'].to_numpy(), 
        dims = 'obs_id'
    )

    gp_sigma = pm.Exponential('gps_sigma', 5)
    ls = pm.InverseGamma(
        'ls',
        alpha = between_season_gp.alpha,
        beta = between_season_gp.beta
    )
    cov_coach_evo = gp_sigma**2*pm.gp.cov.Matern52(input_dim=1, ls = ls)
    
    gp_coach_evo = pm.gp.HSGP(
                                m = [between_season_m],
                                c = between_season_c, 
                                cov_func=cov_coach_evo)

    coach_evolution = gp_coach_evo.prior('coach_evolution',
                                        X = x_seasons,
                                        dims = 'seasons')
    
    offset = pm.Normal('coach_offset',
                        mu = -1.3,
                        sigma = 1,
                        dims = ('play_callers', 'play_type'))


    coach_mu_hyper = pm.Normal('coach_mu_hyper',
                                mu = np.array([obs_pass_prior,
                                            obs_rush_prior]),
                                sigma = 0.2,
                                dims = 'play_type')

    coach_sigma = pm.HalfNormal('coach_sigma', sigma = 0.3, dims = 'play_type')

    coach_mean = pm.Deterministic(
        'coach_mean',
        coach_mu_hyper + coach_sigma * offset,
        dims = ('play_callers', 'play_type')
    )

    control_beta = pm.Normal(
        'controls_beta', mu = 0, sigma = 0.05, dims = 'predictors'
    )

    control_contribution = pm.Deterministic(
        'control_contribution', 
        pm.math.dot(control_data, control_beta),
        dims = 'obs_id'
    )

    adstock_alphas = pm.Beta('adstock_alphas', alpha = 1, beta = 8,
                            dims = 'channels')

    adstock_rush_list = []
    for i in range(len(coords_pl['channels'])):
        adstock_rush_list.append(
            geometric_adstock(personnel_rush_data[:, i],
            adstock_alphas[i], l_max = 2)
        )

    
    x_adstock_rush = pt.stack(adstock_rush_list, axis = 1)

    adstock_pass_list = []
    for i in range(len(coords_pl['channels'])):
        adstock_pass_list.append(
            geometric_adstock(personnel_pass_data[:, i],
            adstock_alphas[i], l_max = 2)
        )

    x_adstock_pass = pt.stack(adstock_pass_list, axis = 1)

    mm_alpha = pm.Gamma('mm_alpha', alpha = 2, beta = 4,
                        dims = ('channels', 'play_type'))
    mm_lam = pm.Gamma('mm_lam', alpha = 2, beta = 10,
                        dims = ('channels', 'play_type'))

    saturated_rush_list = []

    for i in range(len(coords_pl['channels'])):
        saturated_rush_list.append(
            michaelis_menten(x_adstock_rush[:, i], 
            mm_alpha[i,1], mm_lam[i,1])
        )

    saturated_pass_list = []
    for i in range(len(coords_pl['channels'])):
        saturated_pass_list.append(
            michaelis_menten(x_adstock_pass[:, i], 
            mm_alpha[i,0], mm_lam[i,0])
        )



    x_saturated_rush = pt.stack(saturated_rush_list, axis = 1)
    x_saturated_pass = pt.stack(saturated_pass_list, axis = 1)

    personnel_contribution_rush = pm.Deterministic(
        'personnel_contribution_rush',
        x_saturated_rush.sum(axis = 1),
        dims = 'obs_id'
    )
    personnel_contribution_pass = pm.Deterministic(
        'personnel_contribution_pass',
        x_saturated_pass.sum(axis = 1),
        dims = 'obs_id'
    )

    mu_pass = pm.Deterministic(
        'mu_pass',
        coach_mean[coach_idx, 0]  
        + coach_evolution[seasons_idx] 
        + personnel_contribution_pass
        + control_contribution,
        dims = 'obs_id' 

    )

    mu_rush = pm.Deterministic(
        'mu_rush', 
        coach_mean[coach_idx, 1]
        + coach_evolution[seasons_idx]
        + personnel_contribution_rush
        + control_contribution, 
        dims = 'obs_id'
    )

    w_intercept = pm.Normal('w_intercept', mu = 0, sigma = 0.5)
    w_betas = pm.Normal('w_betas',
                        mu = np.array([
                        0.5,
                        0.3,
                        0.5
                        ]),
                        sigma = 0.2,
                        dims = 'weight_predictors')
    
    w_pass_logit = pm.Deterministic(
        'w_pass_logit',
        w_intercept + pm.math.dot(weights_data, w_betas), 
        dims = 'obs_id'
    )

    w_pass = pm.math.invlogit(w_pass_logit)
    w_rush = 1 - w_pass

    w_obs = pm.Deterministic(
        'w_obs', 
        pt.stack([w_pass, w_rush], axis =1), 
        dims = ('obs_id', 'play_type')
    )
    precision_pass = pm.Exponential('precision_pass', 1/20)
    precision_rush = pm.Exponential('precision_rush', 1/30) 


    spread_spline = pm.Data(
        'spread_spline',
        spread_basis,
        dims=('obs_id','spread_basis')
    )

    total_line_spline = pm.Data(
        'total_spline',
        total_basis,
        dims = ('obs_id', 'total_basis')
    )

    w_rush_zero_int = pm.Normal('w_rush_int', -2, 0.5)
    w_rush_coefs_sp = pm.Normal('rush_coefs_sp', 0, 0.3, dims = 'spread_basis')
    w_rush_coefs_tot = pm.Normal('rush_coefs_tot', 0, 0.3, dims = 'total_basis')

    w_rush_zero_logit = pm.Deterministic(
        'w_rush_zero_logit',
        w_rush_zero_int + 
        pm.math.dot(spread_spline, w_rush_coefs_sp) + 
        pm.math.dot(total_line_spline, w_rush_coefs_tot),
        dims = 'obs_id'
    )

    w_rush_zero_obs = pm.math.invlogit(w_rush_zero_logit)

    w_pass_zero_int = pm.Normal('w_pass_int', -1.5, 0.5)
    w_pass_coefs_sp = pm.Normal('pass_coefs_sp', 0, 0.3, dims = 'spread_basis')
    w_pass_coefs_tot = pm.Normal('pass_coefs_tot', 0, 0.3, dims = 'total_basis')

    w_pass_zero_logit = pm.Deterministic(
        'w_pass_zero_logit',
        w_pass_zero_int + 
        pm.math.dot(spread_spline, w_pass_coefs_sp) + 
        pm.math.dot(total_line_spline, w_pass_coefs_tot),
        dims = 'obs_id'
    )

    w_pass_zero_obs = pm.math.invlogit(w_pass_zero_logit)

    w_rush_inner = pt.stack([w_rush_zero_obs, 
                            1 - w_rush_zero_obs], axis=1)

    w_pass_inner = pt.stack([w_pass_zero_obs,
                            1 - w_pass_zero_obs], axis=1)
    
    zero_spike = pm.Beta.dist(alpha = 0.5, beta = 20)

    rush_dist = pm.Mixture.dist(
        w = w_rush_inner,
        comp_dists=[
            zero_spike,
            pm.Beta.dist(mu = pm.math.invlogit(mu_rush), nu = precision_rush)
        ]
    )

    pass_dist = pm.Mixture.dist(
        w = w_pass_inner,
        comp_dists = [
            zero_spike,
            pm.Beta.dist(mu = pm.math.invlogit(mu_pass), nu = precision_pass)
        ]
    )

    # Outer mixture — overall rate
    pm.Mixture(
        'overall_explosive_play_rate',
        w=w_obs,
        comp_dists=[pass_dist, rush_dist],
        observed=overall_exp,
        dims='obs_id'
    )

    pm.Mixture(
        'obs_rush',
        w = w_rush_inner,
        comp_dists = [
            pm.Beta.dist(alpha = 0.5, beta = 20),
            pm.Beta.dist(mu = pm.math.invlogit(mu_rush), nu = precision_rush)
        ],
        observed=rush_obs,
        dims = 'obs_id'
    )

    pm.Mixture(
        'obs_pass',
        w = w_pass_inner,
        comp_dists = [
            pm.Beta.dist(alpha = 0.5, beta = 20),
            pm.Beta.dist(mu = pm.math.invlogit(mu_pass), nu = precision_pass)
        ],
        observed= pass_obs,
        dims = 'obs_id'
    )


    



with mmm_pass_rush:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, group = 'prior', observed=True)


with mmm_pass_rush:
    idata.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )


with mmm_pass_rush:
    idata.extend(
        pm.sample_posterior_predictive(idata, compile_kwargs={'mode': "NUMBA"})
    )

idata.to_
