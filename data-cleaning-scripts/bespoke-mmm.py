import preliz as pz 
import pymc as pm 
from pymc_marketing.mmm.transformers import geometric_adstock
import matplotlib.pyplot as plt 
import polars as pl
import pytensor.tensor as pt
import polars.selectors as cs
import pandas as pd 
import arviz as az
import numpy as np
import seaborn as sns

seed = 14993111

RANDOM_SEED = np.random.default_rng(seed = seed)

raw_data = (
    pl.read_parquet('processed-data/processed-dat.parquet')
    .with_columns(
        pl.col('nflverse_game_id')
        .str.extract(r"_(\d{2})_")
        .str.replace_all('_', '')
        .str.to_integer()
        .alias('week')
    )   
)

raw_data.filter(
    (pl.col("season") == 2017) &
    (pl.col('off_play_caller') == 'Andy Reid')
).select(pl.col('play_caller_tenure'))

raw_data_pd = raw_data.to_pandas()


unique_seasons = raw_data_pd['play_caller_tenure'].sort_values().unique()
unique_play_callers = raw_data_pd['off_play_caller'].sort_values().unique()

season_idx = pd.Categorical(
    raw_data_pd['play_caller_tenure'], categories=unique_seasons
).codes

unique_games = raw_data_pd['week'].sort_values().unique()
games_idx = pd.Categorical(
    raw_data_pd['week'], categories=unique_games
).codes

coach_idx = pd.Categorical(
    raw_data_pd['off_play_caller'], categories=unique_play_callers
).codes



check = raw_data.with_columns(
    pl.col('off_play_caller') == 'Kyle Shanahan'
)

g = sns.FacetGrid(data = check, col = 'season')

g.map(sns.barplot,'week','explosive_play_rate')

## lets get the inseason explosive play rate gp 
## This feels like the sort of stickiness that we are looking for 
## if you get a play calling advantage in general it is not going to last all that long 







predictors = ['avg_epa', 'avg_defenders_in_box', 'roof', 'surface', 'div_game',
                'wind', 'temp', 'is_home_team', 'avg_diff']


just_controls = raw_data_pd[predictors]




epsilon = 0.001
raw_data_pd['explosive_play_rate'] = (
    raw_data_pd['explosive_play_rate']
    .clip(epsilon, 1 - epsilon)
)

y_clipped = raw_data_pd['explosive_play_rate']

## no we are going to set the deviations from our explosive plays
## the global deviation is effectively 3 percent which is pretty big! 



global_sigma,_ = pz.maxent(pz.Exponential(), lower = 0.01, upper = 0.16)


plt.subplot()

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
            pl.exclude('div_game', 'is_home_team', 'surface', 'roof')).columns
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
        pl.Series('roof', just_controls_pl['roof']), 
        pl.Series ('surface', just_controls_pl['surface'])
    ).to_pandas()
)

add_dummies = pd.get_dummies(
    just_controls_sdz, 
    columns = ['roof', 'surface'],
    drop_first=True
).astype(float)


personnel_cols = (
    raw_data.select(cs.starts_with('personnel')).to_pandas()
)

games_x = unique_games- unique_games.min()
seasons_x = unique_seasons - unique_seasons.min()


fig, ax = plt.subplots()

## this looks a bit more skeptical just because we are accounting 
## for roster age, injury, etc
between_season_gp, _ = pz.maxent(pz.InverseGamma(), 2, 3)


raw_data_pd['play_caller_tenure'].mean()

between_season_m, between_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [unique_seasons.min(), unique_seasons.max()],
    lengthscale_range=[2,5],
    cov_func='matern52'

)

fig, ax = plt.subplots()

in_season_gp, _ =  pz.maxent(pz.InverseGamma(), 2, 4)

in_season_m, in_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range=[unique_games.min(), unique_games.max()], 
    lengthscale_range=[2,4], 
    cov_func='matern52'
)


coords = {
    'games': unique_games,
    'seasons': unique_seasons, 
    'predictors': add_dummies.columns.tolist(), 
    'channels': personnel_cols.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index,
    'time_scale': ['season', 'games']
}
def logit(p):
    # Clips p to avoid log(0) errors
    p = np.clip(p, 0.001, 0.999)
    return np.log(p / (1 - p))


x_games_val = games_x.flatten() 
x_seasons_val = seasons_x.flatten()


with pm.Model(coords = coords) as mmm: 

    global_controls = pm.Data(
        'control_data', add_dummies, dims = ('obs_id', 'predictors')
    )
    personnel_dat = pm.Data(
        'personel_data', personnel_cols, dims = ('obs_id','channels')
    )


    games_id = pm.Data('games_id', games_idx, dims = 'obs_id')
    season_id = pm.Data('season_id', season_idx, dims = 'obs_id')

    obs_exp_plays = pm.Data('obs_exp_plays', y_clipped, dims = 'obs_id')

    x_games_data = pm.Data('x_games_data', unique_games,
                            dims='games')[:, None]
    x_seasons_data = pm.Data('x_seasons_data',
                            unique_seasons, dims='seasons')[:, None]

    # super vague 
    gps_sigma = pm.HalfNormal('gps_sigma', 0.1, dims = 'time_scale')
    ls = pm.InverseGamma('ls',
                        alpha = np.array([between_season_gp.alpha,
                                            in_season_gp.alpha]), 
                        beta = np.array([between_season_gp.beta,
                                        in_season_gp.beta]),
                        dims = 'time_scale')

    cov_games = gps_sigma[0]**2 * pm.gp.cov.Matern52(input_dim=1, ls = ls[0])
    cov_seasons = gps_sigma[1]**2 * pm.gp.cov.Matern52(input_dim=1, ls = ls[1])

    gp_games = pm.gp.HSGP(m = [in_season_m], c = in_season_c, 
    cov_func=cov_games)
    gp_season = pm.gp.HSGP(m = [between_season_m], c= between_season_c, 
    cov_func=cov_seasons)

    f_games = gp_games.prior('f_games', X = x_games_data, dims = 'games')
    f_seasons = gp_season.prior('f_seasons', X = x_seasons_data, dims = 'seasons')

    ## now add the coach effects 
    coach_sigma = global_sigma.to_pymc('coach_sigma')
    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = 0,
                                sigma = 1,
                                dims = 'play_callers')
    

    coach_effect = pm.Deterministic('coach_effect',
                        coach_mean_raw * coach_sigma,
                        dims = 'play_callers')

    covariates_prior = pm.Normal('covariates_prior',
                                sigma = 0.25,
                                dims = 'predictors')
    
    
    coach_mu = pm.Deterministic('coach_mu', coach_effect[coach_idx] + 
                                f_games[games_idx] + 
                                f_seasons[season_idx],
                                dims = 'obs_id')
    
    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 2,
                                beta = 2, dims = 'channels')
    
    adstock_list = []
    for i in range(len(coords['channels'])):
        cols = personnel_dat[:,i]
        adstock_list.append(geometric_adstock(cols, adstock_alphas[i],
                            l_max = 3))


    x_adstock = pt.stack(adstock_list, axis = 1)
    
    channel_betas = pm.Normal('channel_betas', mu = 0, sigma = 0.1,
                                dims = 'channels')
    
    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        pm.math.dot(x_adstock, channel_betas),
        dims = 'obs_id'
    )



    global_explosive = pm.Normal('global_effect',
                                mu = -2.1,
                                sigma = 0.25)
    
    mu_logit = pm.Deterministic(
        'mu_logit', 
         coach_mu
        + global_explosive
        + pm.math.dot(global_controls, covariates_prior)
        + personnel_contribution
    )

    mu = pm.Deterministic('mu', pm.math.invlogit(mu_logit))

    precision = pm.Gamma('precision', 20, 1)

    pm.Beta('y_obs', mu = mu, nu = precision,
            observed=obs_exp_plays, dims ='obs_id')



with mmm:
    idata = pm.sample_prior_predictive()

plt.subplots()
az.plot_ppc(idata,group = 'prior', observed=True)






## lets see who struggles and then we can start tuning those parameters 

with mmm:
    idata.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )

az.plot_trace(
    idata
)

check_ess = az.ess(idata.posterior)


az.plot_ess(idata, kind = 'evolution',
            var_names=[RV.name for RV in mmm.free_RVs if RV.size.eval() <= 5])





## so the key issue is that the play designs forsure change
## new motions get implemented and who is on the field changes 
## Since kyle has gotten CMC he is generally in a lot more 2 rb sets 
## whereas a bit earlier in his career he was way more of an 11 guy
## In effect if we are treating the personnel groupings as channels 
## then 