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
    'obs_id': just_controls_sdz.index, 
    'time_scale': ['game', 'season']
}

def logit(p):
    return np.log(p / (1 - p))



with pm.Model(coords = coords) as mmm_hsgp: 

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

    game_data = pm.Data('game_id', games_idx, dims = 'obs_id' )

    x_seasons = pm.Data('x_seasons', unique_seasons, dims = 'seasons')[:,None]

    x_games = pm.Data('x_games', unique_games, dims = 'games')[:, None]

    obs_exp_plays = pm.Data('obs_exp_plays',
                            scaled_y['explosive_play_rate_transformed'].to_numpy(),
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
                                mu = logit(obs_exp_plays.mean()),
                                sigma = 0.5,
                                dims = 'play_callers')

    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean_raw[coach_idx] + 
                                coach_evolution[season_idx] +
                                game_evolution[games_idx],
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

    channel_betas = pm.Normal('channel_betas', mu = 0, sigma = 0.5,
                                dims = 'channels')

    saturation_lam = pm.Gamma('saturation_lamda',
                                                alpha = 3,
                                                beta = 0.5, dims = 'channels')

    saturation_slope = pm.LogNormal('saturation_slope',
                                        mu = 0,
                                        sigma = 0.2, 
                                        dims = 'channels')
    # effectivelly logistic saturation
    x_saturated = pm.Deterministic(
                'x_saturated',
                ((1-pm.math.exp(-saturation_lam * x_adstock)) / 
                (1 + pm.math.exp(-saturation_lam * x_adstock))),
                dims = ('obs_id', 'channels'))


    personnel_contribution = pm.Deterministic(
        'personnel_contribution', 
        pm.math.dot(x_saturated, channel_betas),
        dims = 'obs_id'

    )

    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.5)



    mu = pm.Deterministic('mu',
        coach_mu +  
        personnel_contribution +  
        control_contribution + 
        (pm.math.dot(passing_prior, passing_dat)),      
        dims='obs_id'
    )
    # like 1/25 plays is explosive
    precision = pm.Exponential('precision', 1/20)
    #precision  = pm.Gamma('precision', alpha = 10, beta = 1)
    mu_logit = pm.math.invlogit(mu)
  

    pm.Beta(
        'y_obs', 
        alpha = mu_logit * precision, 
        beta = (1-mu_logit) * precision,
        observed = obs_exp_plays,
        dims = 'obs_id')


with mmm_hsgp:
    idata = pm.sample_prior_predictive()

az.plot_ppc(idata, group = 'prior', observed=True)


with mmm_hsgp:
    idata.extend(
        pm.sample(
            random_seed=RANDOM_SEED, nuts_sampler = 'numpyro'
        )
    )


with mmm_hsgp: 
    pm.compute_log_likelihood(idata)
    idata.extend(
        pm.sample_posterior_predictive(idata),
    )

az.plot_ppc(idata)


idata.to_netcdf('model/mmm-beta.nc')

plt.title('Regular Beta')


post = idata['posterior']

shanny_mean = post['coach_mean_raw'].sel(play_callers = 'Kyle Shanahan')

shanny_idx = list(coords['play_callers']).index('Kyle Shanahan')
shanny_mask = coach_idx == shanny_idx

shanahan_obs = np.where(shanny_mask)[0]
base = post["coach_mu"].isel(obs_id=shanahan_obs).mean(dim=["chain", "draw"]).values
per_channel = (post["x_saturated"].isel(obs_id=shanahan_obs) * post["channel_betas"]).mean(dim=["chain", "draw"]).values  
controls    = post["control_contribution"].isel(obs_id=shanahan_obs).mean(dim=["chain", "draw"]).values
base_layer     = expit(base)
channel_layers = []
cumulative     = base.copy()

for i in range(len(coords["channels"])):
    cumulative = cumulative + per_channel[:, i]
    channel_layers.append(expit(cumulative))

observed = scaled_y["explosive_play_rate_transformed"].to_numpy()[shanny_mask]
games   = np.arange(len(shanahan_obs))  

fig, ax = plt.subplots(figsize=(14, 6))

colors = plt.cm.tab10.colors

ax.fill_between(games, 0, base_layer, color="gray", alpha=0.8, label="Base")

prev_layer = base_layer.copy()
for i, channel in enumerate(coords["channels"]):
    ax.fill_between(games, prev_layer, channel_layers[i],
                    color=colors[i], alpha=0.8, label=channel)
    prev_layer = channel_layers[i]

ax.plot(games, observed, color="black", linewidth=1, label="Observed")

ax.set_title("Personnel Contribution Breakdown — Kyle Shanahan")
ax.set_xlabel("Game")
ax.set_ylabel("Explosive Play Rate")
ax.set_ylim(0, 1)  # probability scale
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
## vanishingly small contribution 

az.plot_posterior(idata, var_names=["channel_betas", "saturation_lamda", "saturation_slope"])

x_sat_vals = post["x_saturated"].mean(dim=["chain","draw"]).values  # (obs, channels)
for i, ch in enumerate(coords["channels"]):
    print(f"{ch}: min={x_sat_vals[:,i].min():.3f}, max={x_sat_vals[:,i].max():.3f}, std={x_sat_vals[:,i].std():.3f}")





