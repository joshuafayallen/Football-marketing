from modeling-scripts.bespoke-mmm import RANDOM_SEED
import preliz as pz 
import pymc as pm 
from pymc_marketing.mmm.transformers import geometric_adstock, hill_function, logistic_saturation
import matplotlib.pyplot as plt 
import polars as pl
import pytensor.tensor as pt
import polars.selectors as cs
import pandas as pd 
from sklearn.preprocessing import MaxAbsScaler
import arviz as az
import numpy as np
import seaborn as sns

seed = 14993111

RANDOM_SEED = np.random.default_rng(seed = seed)

def plot_prior(trace ,param: str):
    fig, axe = plt.subplots()
    return az.plot_dist(trace.prior[param])



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
            'wind', 'temp', 'is_home_team', 'avg_diff', ]





just_controls = raw_data_pd[predictors]

nobs = raw_data.height

scaled_y = (
    raw_data
    .with_columns(
    (pl.col('explosive_play_rate') / (pl.col('explosive_play_rate').abs().max())).alias("scaled_explosive_plays")
    ).select(
        (((pl.col('scaled_explosive_plays')) * (nobs-1) + 0.5)/nobs).alias("explosive_play_rate_scaled")
    )

)

fig,axe = plt.subplots()

plt.hist(scaled_y['explosive_play_rate_scaled'])

scaled_y['explosive_play_rate_scaled'].max()



## no we are going to set the deviations from our explosive plays

## the global deviation is effectively 3 percent which is pretty big! 







global_sigma,_ = pz.maxent(pz.Exponential(), lower = 0.01, upper = 0.16)





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


usage = personnel_cols.mean(axis = 0)

reliable_cols = usage[usage >= 0.01].index.tolist()

personnel_cols = personnel_cols[reliable_cols]

reliable_cols


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
    'predictors': just_controls_sdz.columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index, 
    'time_scale': ['game', 'season']
}




with pm.Model(coords = coords) as mmm_hsgp: 

    global_controls = pm.Data(
        'control_data', just_controls_sdz, dims = ('obs_id', 'predictors')
    )

    personnel_dat = pm.Data(
        'personel_data', personnel_scaled, dims = ('obs_id','channels')
    )

    season_data = pm.Data('season_id', season_idx, dims = 'obs_id')

    game_data = pm.Data('game_id', games_idx, dims = 'obs_id' )

    x_seasons = pm.Data('x_seasons', unique_seasons, dims = 'seasons')[:,None]

    x_games = pm.Data('x_games', unique_games, dims = 'games')[:, None]

    obs_exp_plays = pm.Data('obs_exp_plays',
                            scaled_y['explosive_play_rate_scaled'].to_numpy(),
                            dims = 'obs_id')

    #
    gps_sigma = pm.Exponential('gps_sigma', 10, dims='time_scale')

    ls = pm.InverseGamma('ls',
                        alpha = np.array([in_season_gp.alpha,
                                        between_season_gp.alpha]),
                        beta = np.array([in_season_gp.beta,
                                        between_season_gp.beta]),
                        dims='time_scale')


    cov_coach_evo = gps_sigma[0]**2*pm.gp.cov.Matern52(input_dim=1, ls = ls[0])

    cov_game_evo = gps_sigma[1]**2*pm.gp.cov.Matern52(input_dim=1, ls = ls[1])


    gp_coach_evo = pm.gp.HSGP(m = [between_season_m],
                            c = between_season_c,
                            cov_func=cov_coach_evo)

    gp_game_evo = pm.gp.HSGP(m = [in_season_m],
                            c = in_season_c,
                            cov_func=cov_game_evo)

    coach_evolution = gp_coach_evo.prior('coach_evolution',
                                            X = x_seasons,
                                            dims = 'seasons')


    game_evolution = gp_game_evo.prior('game_evolution',
                                        X = x_games,
                                        dims = 'games')

    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = 0,
                                sigma = 0.2,
                                dims = 'play_callers')


    covariates_prior = pm.Normal('covariates_prior',
                                sigma = 0.2,
                                dims = 'predictors')



    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean_raw[coach_idx] + 
                                coach_evolution[season_idx] +
                                game_evolution[games_idx],
                                dims = 'obs_id')

                                

    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 1,
                                beta = 8, dims = 'channels')



    controls_beta = pm.Normal('controls_beta', mu=0, sigma= 0.1,

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
                            l_max = 1))

    x_adstock = pt.stack(adstock_list, axis = 1)

    channel_betas = pm.Gamma('channel_betas', alpha = 2, beta = 20,
                                dims = 'channels')

    saturation_lam = pm.Gamma('saturation_lamda',
                                                alpha = 3,
                                                beta = 0.5, dims = 'channels')

    saturation_slope = pm.LogNormal('saturation_slope',
                                        mu = 0,
                                        sigma = 0.2, 
                                        dims = 'channels')

    x_saturated = ((1-pm.math.exp(-saturation_lam * x_adstock)) / (1 + pm.math.exp(-saturation_lam * x_adstock)))


    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        pm.math.dot(x_saturated, channel_betas),
        dims = 'obs_id'

    )

    mu_logit = pm.Deterministic('mu_logit',
        pm.math.invlogit( 
        coach_mu +     
        personnel_contribution +  
        control_contribution),      
        dims='obs_id'
    )
    # like 1/25 plays is explosive
    precision = pm.Exponential('precision', 1/20)
    # precision  = pm.Gamma('precision', alpha = 20, beta = 1)
  

    pm.Beta(
        'y_obs', 
        mu = mu_logit, 
        nu = precision,
        observed = obs_exp_plays,
        dims = 'obs_id')


with mmm_hsgp:
    idata = pm.sample_prior_predictive()


az.plot_ppc(idata, group = 'prior', observed=True)


plot_prior(idata, param = 'mu_logit')

plot_prior(idata, param = 'precision')
plot_prior(idata, 'coach_mean_raw')
plot_prior(idata, 'covariates_prior')
plot_prior(idata, 'adstock_alphas')


with mmm_hsgp:
    idata.extend(
        pm.sample(
            random_seed=RANDOM_SEED, nuts_sampler = 'numpyro'
        )
    )


az.plot_ess(
    idata,
    kind = 'evolution',
    var_names=[RV.name for RV in mmm_hsgp.free_RVs if RV.size.eval() <= 3]
)

fig,axes = plt.subplots()
pz.Gamma(3, 0.5).plot_pdf()

with mmm_hsgp: 
    idata.extend(
        pm.sample_posterior_predictive(idata)
    )

az.plot_ppc(idata)
