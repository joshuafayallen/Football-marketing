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
from scipy.special import expit
import seaborn as sns

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
        pl.col('games_called') >= 100 # generally we want at least like two years of data 
    )
    ['off_play_caller'].to_list()
)

check_py = (pl.read_parquet('processed-data/processed-dat.parquet')
    .group_by('off_play_caller')
    .agg(
        pl.len().alias('games_called')
    )
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
        .alias('is_indoors')    
            )
        .filter(
            pl.col('off_play_caller').is_in(keep_these)
        )
)   





raw_data_pd = raw_data.to_pandas()


unique_seasons = raw_data_pd['play_caller_tenure'].sort_values().unique()

unique_play_callers = raw_data_pd['off_play_caller'].sort_values().unique()

season_idx = pd.Categorical(
    raw_data_pd['play_caller_tenure'], categories=unique_seasons
).codes


unique_games = raw_data_pd['week'].sort_values().unique()

games_idx = pd.Categorical(
    raw_data_pd['week'], categories=unique_games).codes



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
    (pl.col('explosive_play_rate') / (pl.col('explosive_play_rate').abs().max())).alias("scaled_explosive_plays")
    ).with_columns(
        (((pl.col('scaled_explosive_plays')) * (nobs-1) + 0.5)/nobs).alias("explosive_play_rate_transformed")
    )

)

scaled_y['scaled_explosive_plays'].max()


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
            pl.exclude('div_game', 'is_home_team', 'is_indoors', 'is_grass')).columns
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
just_controls_sdz.columns

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





in_season_gp, _ =  pz.maxent(pz.InverseGamma(), 2, 4)



in_season_m, in_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(

    x_range=[
        unique_games.min(), unique_games.max()
    ], 

    lengthscale_range=[2,4], 
    cov_func='matern52'

)



between_season_gp, _ = pz.maxent(pz.InverseGamma(), 2, 5)


between_season_m, between_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [unique_seasons.min(), unique_seasons.max()],
    lengthscale_range=[2,5],
    cov_func='matern52'

)



coords = {
    'games': unique_games,
    'seasons': unique_seasons, 
    'predictors': just_controls_sdz.drop('avg_pass_rate',axis = 1).columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index
}

def logit(p):
    return np.log(p / (1 - p))


with pm.Model(coords = coords) as mmm_mix: 

    global_controls = pm.Data(
        'control_data', just_controls_sdz.drop('avg_pass_rate', axis = 1),
                        dims = ('obs_id', 'predictors')
    )

    passing_dat = pm.Data('passing_data',
                            just_controls_sdz['avg_pass_rate'],
                            dims = 'obs_id')

    personnel_dat = pm.Data(
        'personel_data', personnel_scaled, dims = ('obs_id','channels')
    )

    season_data = pm.Data('season_id', season_idx, dims = 'obs_id')

    x_seasons = pm.Data('x_seasons', unique_seasons, dims = 'seasons')[:,None]

    obs_exp_plays = pm.Data('obs_exp_plays',
                            scaled_y['explosive_play_rate_transformed'].to_numpy(),
                            dims = 'obs_id')


    gps_sigma = pm.Exponential('gps_sigma', 2)

    ls = pm.InverseGamma('ls',
                        alpha = between_season_gp.alpha,
                        beta = between_season_gp.beta)


    cov_coach_evo = gps_sigma**2*pm.gp.cov.Matern52(input_dim=1, ls = ls)

    gp_coach_evo = pm.gp.HSGP(m = [between_season_m],
                            c = between_season_c,
                            cov_func=cov_coach_evo)


    coach_evolution = gp_coach_evo.prior('coach_evolution',
                                            X = x_seasons,
                                            dims = 'seasons')

    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = logit(0.3),
                                sigma = 0.2,
                                dims = 'play_callers')

    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean_raw[coach_idx] + 
                                coach_evolution[season_idx],
                                dims = 'obs_id')

                                

    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 1,
                                beta = 8, dims = 'channels')

    controls_beta = pm.Normal('controls_beta', mu=0, sigma= 0.05,
    dims='predictors')

    control_contribution = pm.Deterministic(
        'control_contribution', 
        pm.math.dot(global_controls, controls_beta), 
        dims='obs_id'

    )

    adstock_list = []
    

    for i in range(len(coords['channels'])):
        cols = personnel_dat[:,i]
        adstock_list.append(geometric_adstock(cols, adstock_alphas[i],
                            l_max = 2))

    x_adstock = pt.stack(adstock_list, axis = 1)

    mm_alpha = pm.Gamma('mm_alpha', alpha=2, beta=4, dims='channels')   
    mm_lam = pm.Gamma('mm_lam', alpha=2, beta=10, dims='channels')  


    saturated_list = []

    for i in range(len(coords['channels'])):

        saturated_list.append(
        michaelis_menten(x_adstock[:, i], mm_alpha[i], mm_lam[i]))

    x_saturated = pt.stack(saturated_list, axis=1)


    personnel_contribution = pm.Deterministic(
    'personnel_contribution',
    x_saturated.sum(axis=1),  # mm_alpha handles magnitude per channel
    dims='obs_id')
    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)



    mu = pm.Deterministic('mu',
        coach_mu +  
        personnel_contribution +  
        control_contribution + 
        (pm.math.dot(passing_prior, passing_dat)),      
        dims='obs_id'
    )


    mu_logit = pm.math.invlogit(mu)
    # like 1/25 plays is explosive
    w = pm.Dirichlet('w', a=np.array([1, 1, 10])) 

    precision = pm.Exponential('precision', 1/20)

    lower_bound = 0.5 / len(scaled_y)
    upper_bound = 1 - 0.5 / len(scaled_y)

    zero_spike = pm.Uniform.dist(lower=0, upper= lower_bound * 10)
    high_spike = pm.Uniform.dist(lower=1 - (upper_bound * 0.02), upper=1.0)
    beta_dist = pm.Beta.dist(alpha=mu_logit * precision, beta = (1-mu_logit) * precision)

    pm.Mixture('y_obs', w=w, 
                    comp_dists=[zero_spike, high_spike,
                                beta_dist], 
                    observed=obs_exp_plays)


with mmm_mix:
    idata_mix = pm.sample_prior_predictive()

az.plot_ppc(idata_mix, group = 'prior', observed = True)


with mmm_mix:

    idata_mix.extend(
        pm.sample(random_seed=RANDOM_SEED,
                nuts_sampler='numpyro', target_accept = 0.95)
    )


with mmm_mix:
    pm.compute_log_likelihood(idata_mix)
    idata_mix.extend(
        pm.sample_posterior_predictive(idata_mix)
    )


az.plot_ppc(idata_mix)


az.plot_energy(idata_mix)



idata_mix.to_netcdf('model/mmm-mix-fixed-cutpoints.nc')


with pm.Model(coords = coords) as mmm_mix_adapt: 

    global_controls = pm.Data(
        'control_data', just_controls_sdz.drop('avg_pass_rate', axis = 1),
                        dims = ('obs_id', 'predictors')
    )

    passing_dat = pm.Data('passing_data',
                            just_controls_sdz['avg_pass_rate'],
                            dims = 'obs_id')

    personnel_dat = pm.Data(
        'personel_data', personnel_scaled, dims = ('obs_id','channels')
    )

    season_data = pm.Data('season_id', season_idx, dims = 'obs_id')

    x_seasons = pm.Data('x_seasons', unique_seasons, dims = 'seasons')[:,None]

    obs_exp_plays = pm.Data('obs_exp_plays',
                            scaled_y['explosive_play_rate_transformed'].to_numpy(),
                            dims = 'obs_id')


    gps_sigma = pm.Exponential('gps_sigma', 2)

    ls = pm.InverseGamma('ls',
                        alpha = between_season_gp.alpha,
                        beta = between_season_gp.beta)


    cov_coach_evo = gps_sigma**2*pm.gp.cov.Matern52(input_dim=1, ls = ls)

    gp_coach_evo = pm.gp.HSGP(m = [between_season_m],
                            c = between_season_c,
                            cov_func=cov_coach_evo)


    coach_evolution = gp_coach_evo.prior('coach_evolution',
                                            X = x_seasons,
                                            dims = 'seasons')

    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = logit(0.3),
                                sigma = 0.2,
                                dims = 'play_callers')

    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean_raw[coach_idx] + 
                                coach_evolution[season_idx],
                                dims = 'obs_id')

                                

    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 1,
                                beta = 8, dims = 'channels')

    controls_beta = pm.Normal('controls_beta', mu=0, sigma= 0.05,
    dims='predictors')

    control_contribution = pm.Deterministic(
        'control_contribution', 
        pm.math.dot(global_controls, controls_beta), 
        dims='obs_id'

    )

    adstock_list = []
    

    for i in range(len(coords['channels'])):
        cols = personnel_dat[:,i]
        adstock_list.append(geometric_adstock(cols, adstock_alphas[i],
                            l_max = 2))

    x_adstock = pt.stack(adstock_list, axis = 1)

    mm_alpha = pm.Gamma('mm_alpha', alpha=2, beta=4, dims='channels')   
    mm_lam = pm.Gamma('mm_lam', alpha=2, beta=10, dims='channels')  


    saturated_list = []

    for i in range(len(coords['channels'])):

        saturated_list.append(
        michaelis_menten(x_adstock[:, i], mm_alpha[i], mm_lam[i]))

    x_saturated = pt.stack(saturated_list, axis=1)


    personnel_contribution = pm.Deterministic(
    'personnel_contribution',
    x_saturated.sum(axis=1),  # mm_alpha handles magnitude per channel
    dims='obs_id')
    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)



    mu = pm.Deterministic('mu',
        coach_mu +  
        personnel_contribution +  
        control_contribution + 
        (pm.math.dot(passing_prior, passing_dat)),      
        dims='obs_id'
    )


    mu_logit = pm.math.invlogit(mu)
    # like 1/25 plays is explosive
    precision = pm.Exponential('precision', 1/20)
    #precision = pm.Gamma('precision', alpha = 10, beta = 0.5)

    # set hyper priors for cutpoints 

    w_mu = pm.Normal('w_mu', mu = 0, sigma = 0.5)
    w_sig = pm.HalfNormal('w_sig', sigma = 0.3)

    # the offset is probably not huge
    w_coach_offset = pm.Normal('w_coach_offset',
                                mu = 0,
                                sigma = 1,
                                dims = 'play_callers')
    
    w_coach_logit = pm.Deterministic(
        'w_coach_logit', 
        w_mu + w_sig * w_coach_offset,
        dims = 'play_callers'
    )

    w_zeros = pm.math.invlogit(w_coach_logit[coach_idx])

    w_beta_obs = 1 - w_zeros

    w_obs = pm.Deterministic(
        'w_obs',
        pt.stack([w_zeros, w_beta_obs], axis = 1)
    )
    lower_bound = 0.5 / len(scaled_y)
    upper_bound = 1 - 0.5 / len(scaled_y)

    zero_spike = pm.Uniform.dist(lower = 0, upper=lower_bound * 10)

    beta_dist = pm.Beta.dist(alpha=mu_logit * precision,
                            beta = (1-mu_logit) * precision)
    pm.Mixture('y_obs',
                    w=w_obs, 
                    comp_dists=[zero_spike, 
                                beta_dist], 
                    observed=obs_exp_plays)

    pm.Mixture('y_obs',
                    w=w_obs, 
                    comp_dists=[zero_spike, 
                                beta_dist], 
                    observed=obs_exp_plays)



with mmm_mix_adapt:
    idata_adapt_mix = pm.sample_prior_predictive()


az.plot_ppc(idata_adapt_mix, group = 'prior', observed = True)

with mmm_mix_adapt:
    idata_adapt_mix.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )



with mmm_mix_adapt:
    pm.compute_log_likelihood(idata_adapt_mix)
    idata_adapt_mix.extend(
        pm.sample_posterior_predictive(idata_adapt_mix)
    )


idata_adapt_mix.to_netcdf(
    'model/mmm-mix-adapt-cutpoints.nc'
)

az.plot_ppc(idata_adapt_mix)


plt.close("all")


az.plot_energy(idata_adapt_mix)
