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
from patsy import dmatrix
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
        pl.col('games_called') >= 104 # generally we want at least like two years of data 
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





between_season_gp, _ = pz.maxent(pz.InverseGamma(), 2, 3)


between_season_m, between_season_c = pm.gp.hsgp_approx.approx_hsgp_hyperparams(
    x_range = [unique_seasons.min(), unique_seasons.max()],
    lengthscale_range=[2,3],
    cov_func='matern52'

)

def logit(p):
    return np.log(p / (1 - p))

coords = {
    'games': unique_games,
    'seasons': unique_seasons, 
    'predictors': just_controls_sdz.drop('avg_pass_rate',axis = 1).columns.tolist(), 
    'channels': personnel_scaled.columns,
    'play_callers': unique_play_callers,
    'obs_id': just_controls_sdz.index, 
}


with pm.Model(coords = coords) as sans_games_mm:
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


    gps_sigma = pm.Exponential('gps_sigma', 10)

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

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5)
    coach_raw = pm.Normal('coach_raw', 0,1, dims = 'play_callers')

    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = logit(obs_exp_plays.mean()),
                                sigma = 0.5)
    
    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_mean_raw + coach_sigma * coach_raw, 
        dims = 'play_callers'
    )
    

    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean[coach_idx] + 
                                coach_evolution[season_idx],
                                dims = 'obs_id')

                                

    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 1,
                                beta = 12, dims = 'channels')

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

    x_adstock = pm.Deterministic(
        'x_adstock',
        pt.stack(adstock_list, axis = 1),
        dims = ('obs_id', 'channels'))

    mm_alpha = pm.Gamma('mm_alpha', alpha=3, beta=6, dims='channels')   
    mm_lam = pm.Gamma('mm_lam', alpha=3, beta=15, dims='channels')  


    saturated_list = []

    for i in range(len(coords['channels'])):

        saturated_list.append(
        michaelis_menten(x_adstock[:, i], mm_alpha[i], mm_lam[i]))

    x_saturated = pm.Deterministic(
        'x_saturated',
        pt.stack(saturated_list, axis=1),
        dims = ('obs_id', 'channels'))



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
    # like 1/25 plays is explosive
    precision = pm.Exponential('precision', 1/20)
    #precision = pm.Gamma('precision', alpha = 10, beta = 0.5)
    mu_logit = pm.math.invlogit(mu)


    pm.Beta(
        'y_obs', 
        mu = mu_logit , 
        nu =  precision,
        observed = obs_exp_plays,
        dims = 'obs_id')
    



with sans_games_mm:
    idata_hsgp = pm.sample_prior_predictive()


az.plot_ppc(idata_hsgp, group = 'prior', observed = True)


with sans_games_mm:
    idata_hsgp.extend(
        pm.sample(
            random_seed=RANDOM_SEED, nuts_sampler='numpyro', progressbar=True
        )
    )


with sans_games_mm:
    idata_hsgp.extend(
        pm.sample_posterior_predictive(idata_hsgp)
    )

az.plot_ppc(idata_hsgp)

az.plot_energy(idata_hsgp)
len(unique_seasons) / 3

n_knots = 9
knots = np.quantile(unique_seasons, np.linspace(0,1, n_knots))

spline = dmatrix(
    "bs(tenure, knots = knots, degree = 3, include_intercept = True) - 1", 
    {'tenure': unique_seasons, "knots": knots[1:-1]}
)

basis_set = np.array(spline)

coords['spline_basis'] = [f"s{i}" for i in range(basis_set.shape[1])]


with pm.Model(coords = coords) as mmm_spline:
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

    spline_basis_dat = pm.Data(
        'spline_bais', 
        basis_set[season_idx], 
        dims = ('obs_id', 'spline_basis')
    )

    obs_exp_plays = pm.Data('obs_exp_plays',
                            scaled_y['explosive_play_rate_transformed'].to_numpy(),
                            dims = 'obs_id')
    
    spline_sigma = pm.Exponential('beta_sigma', 2)
    spline_raw = pm.Normal('spline_raw', 0 , 1, shape = basis_set.shape[1])

    spline_beta = pm.Deterministic(
        'spline_beta', 
        spline_sigma * spline_raw, 
        dims = 'spline_basis'
    )

    season_contributions = pm.Deterministic(
        'season_contribution', 
        pm.math.dot(spline_basis_dat, spline_beta), 
        dims = 'obs_id'
    )

    coach_sigma = pm.HalfNormal('coach_sigma', 0.5)
    coach_raw = pm.Normal('coach_raw', 0,1, dims = 'play_callers')

    coach_mean_raw = pm.Normal('coach_mean_raw',
                                mu = logit(obs_exp_plays.mean()),
                                sigma = 0.5)
    
    coach_mean = pm.Deterministic(
        'coach_mean', 
        coach_mean_raw + coach_sigma * coach_raw, 
        dims = 'play_callers'
    )
    

    coach_mu = pm.Deterministic('coach_mu', 
                                coach_mean[coach_idx] + 
                                season_contributions,
                                dims = 'obs_id')

                                

    adstock_alphas = pm.Beta('adstock_alphas',
                                alpha = 1,
                                beta = 12, dims = 'channels')

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

    x_adstock = pm.Deterministic(
        'x_adstock',
        pt.stack(adstock_list, axis = 1),
        dims = ('obs_id', 'channels'))

    mm_alpha = pm.Gamma('mm_alpha', alpha=3, beta=6, dims='channels')   
    mm_lam = pm.Gamma('mm_lam', alpha=3, beta=15, dims='channels')  


    saturated_list = []

    for i in range(len(coords['channels'])):

        saturated_list.append(
        michaelis_menten(x_adstock[:, i], mm_alpha[i], mm_lam[i]))

    x_saturated = pm.Deterministic(
        'x_saturated',
        pt.stack(saturated_list, axis=1),
        dims = ('obs_id', 'channels'))

    personnel_contribution = pm.Deterministic(
                                            'personnel_contribution',    
                                            x_saturated.sum(axis=1),
                                            dims='obs_id')


    passing_prior = pm.Normal('passing_prior', mu = 0, sigma = 0.1)



    mu = pm.Deterministic('mu',
        coach_mu +  
        personnel_contribution +  
        control_contribution + 
        (pm.math.dot(passing_prior, passing_dat)),      
        dims='obs_id'
    )
    # like 1/25 plays is explosive
    #precision = pm.Exponential('precision', 1/20)
    precision = pm.Gamma('precision', alpha = 10, beta = 0.5)
    mu_logit = pm.math.invlogit(mu)


    pm.Beta(
        'y_obs', 
        mu = mu_logit , 
        nu = precision,
        observed = obs_exp_plays,
        dims = 'obs_id')

with mmm_spline:
    idata_spline = pm.sample_prior_predictive()


az.plot_ppc(idata_spline, group = 'prior', observed=True)

with mmm_spline:
    idata_spline.extend(
        pm.sample(random_seed=RANDOM_SEED, nuts_sampler='numpyro')
    )



with mmm_spline:
    idata_spline.extend(
        pm.sample_posterior_predictive(idata_spline)
    )

az.plot_energy(idata_spline)

az.plot_ppc(idata_spline)

spline_mu = az.extract(
    idata_spline, group = 'posterior', var_names='spline_beta'
).values

spline_curve = basis_set @ spline_mu

fig,axs = plt.subplots(2,1)

pm.gp.util.plot_gp_dist(
    ax = axs[0], 
    samples = spline_curve.T,
    x = unique_seasons,
    #alpha = 0.5
)

axs[0].set_title('Posterior spline')

hsgp = az.extract(idata_hsgp, group = 'posterior', var_names='coach_evolution')

pm.gp.util.plot_gp_dist(
    ax = axs[1], 
    samples= hsgp.values.T,
    x = unique_seasons,
)

axs[1].set_title('Posterior HSGP')


with sans_games_mm:
    pm.compute_log_likelihood(idata_hsgp)

with mmm_spline:
    pm.compute_log_likelihood(idata_spline)
    

mod_names = ['mod with hsgp', 'mod with spline']

mod_dict = dict(zip(mod_names,[idata_hsgp, idata_spline]))

az.compare(mod_dict)


az.plot_trace(idata_spline, var_names=['coach_mu'])



post_spline = idata_spline['posterior']
shanny_mean = post_spline.sel(play_callers = 'Kyle Shanahan')

shanny_idx = list(coords['play_callers']).index('Kyle Shanahan')

shanny_mask = coach_idx == shanny_idx
shanny_obs = np.where(shanny_mask)[0]


base = post_spline['coach_mu'].isel(obs_id = shanny_obs).mean(dim = ['chain', 'draw']).values

per_channel = (
    post_spline['x_saturated']
    .isel(obs_id = shanny_obs)).mean(dim = ['chain', 'draw']).values 



game_order = np.argsort(shanny_obs)
base = base[game_order]
per_channel = per_channel[game_order, :]

base_layer = expit(base)
channel_layers = []
cumulative = base.copy()


for i in range(len(coords['channels'])): 
    cumulative = cumulative + per_channel[:,i]
    channel_layers.append(expit(cumulative))


observed = scaled_y['explosive_play_rate_transformed'].to_numpy()[shanny_mask]
games = np.arange(len(shanny_obs))

fig, ax = plt.subplots(figsize = (14,6))

colors = plt.cm.tab10.colors


ax.fill_between(games, 0, base_layer, color = 'gray', alpha = 0.5,
                label = 'Base Contribution')

prev_layer = base_layer.copy()

for i, channel in enumerate(coords['channels']):
    ax.fill_between(games, prev_layer, channel_layers[i], 
                    color = colors[i], alpha = 0.6, label = channel)
    prev_layer = channel_layers[i]

ax.plot(games, observed, color = 'black', linewidth = 1, label = 'Observed')
ax.set_xlabel('Games')
ax.set_ylabel('Explosive Play Rate')
ax.legend(loc = 'upper left', bbox_to_anchor = (1,1))
plt.tight_layout()

mu_samples = az.extract(idata_spline, var_names='mu').values 
x_saturated_samples = az.extract(idata_spline, var_names='x_saturated').values

channel_names = coords['channels']
play_callers = coords['play_callers']

channel_names

records = []

for i, channel in enumerate(channel_names): 
    mu_without = mu_samples - x_saturated_samples[:, i, :]
    marginal = expit(mu_samples) - expit(mu_without)
    for c, caller in enumerate(play_callers):
        caller_mask = coach_idx == c
        if caller_mask.sum() == 0:
            continue
        
        avg = marginal[caller_mask, :].mean(axis = 0)
        hdi = az.hdi(avg, hdi_prob=0.89)
        records.append(
            {
                'play_caller': caller, 
                'channel': channel,
                'mean': avg.mean(), 
                'low': float(hdi[0]), 
                'high': float(hdi[1])
            }
        )


contribution_df = pd.DataFrame(records).sort_values('mean')

just_shanny = contribution_df[contribution_df['play_caller'] == 'Kyle Shanahan']

fig, ax = plt.subplots()

for i, (_, row) in enumerate(just_shanny.iterrows()):
    ax.plot(
        [row['low'], row['high']], [i, i]
    )
    ax.scatter(row['mean'], i)


ax.set_yticks(range(len(just_shanny)))
ax.set_yticklabels(just_shanny['channel'])

ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Average marginal contribution to explosive play rate')
ax.set_title('Kyle Shanahan — personnel marginal contributions')
plt.tight_layout()
plt.show()


mu_samples = az.extract(idata_hsgp, var_names='mu').values 
x_saturated_samples = az.extract(idata_hsgp, var_names='x_saturated').values

channel_names = coords['channels']
play_callers = coords['play_callers']

channel_names

records = []

for i, channel in enumerate(channel_names): 
    mu_without = mu_samples - x_saturated_samples[:, i, :]
    marginal = expit(mu_samples) - expit(mu_without)
    for c, caller in enumerate(play_callers):
        caller_mask = coach_idx == c
        if caller_mask.sum() == 0:
            continue
        
        avg = marginal[caller_mask, :].mean(axis = 0)
        hdi = az.hdi(avg, hdi_prob=0.89)
        records.append(
            {
                'play_caller': caller, 
                'channel': channel,
                'mean': avg.mean(), 
                'low': float(hdi[0]), 
                'high': float(hdi[1])
            }
        )


contribution_df = pd.DataFrame(records).sort_values('mean')

just_shanny = contribution_df[contribution_df['play_caller'] == 'Kyle Shanahan']

fig, ax = plt.subplots()

for i, (_, row) in enumerate(just_shanny.iterrows()):
    ax.plot(
        [row['low'], row['high']], [i, i]
    )
    ax.scatter(row['mean'], i)


ax.set_yticks(range(len(just_shanny)))
ax.set_yticklabels(just_shanny['channel'])

ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_xlabel('Average marginal contribution to explosive play rate')
ax.set_title('Kyle Shanahan — personnel marginal contributions')
plt.tight_layout()
plt.show()


mu_post = (
    az.extract(idata_hsgp, var_names='mu')
    .to_dataframe()
    .reset_index()
)

x_sat_post = (
    az.extract(idata_hsgp, var_names='x_saturated')
    .to_dataframe()
    .reset_index()
)

mm_alpha_post = (
    az.extract(idata_hsgp, var_names=['mm_alpha'])
    .to_dataframe()
    .reset_index()
)
mm_lam_post = (
    az.extract(idata_hsgp, var_names=['mm_lam'])
    .to_dataframe()
    .reset_index()
)

x_adstock_post = (
    az.extract(idata_hsgp, var_names='x_adstock')
    .to_dataframe()
    .reset_index()
)

mm_alpha_post.columns


post_df = (
    mu_post.merge(x_sat_post, on = ['obs_id', 'chain', 'draw'])
    .merge(mm_alpha_post, on  = ['chain', 'draw', 'channels'])
    .merge(mm_lam_post, on = ['chain', 'draw', 'channels'])
    .merge(x_adstock_post, on = ['obs_id', 'chain', 'draw', 'channels'])
)



post_df['play_caller'] = pd.Categorical(
    [play_callers[i] for i in coach_idx[post_df['obs_id'].values]],
    categories = play_callers
)

post_df_hsgp = (pl.from_pandas(
    post_df
)
    .with_columns(
        pl.lit('HSGP Specification').alias('id')
    )
)

mu_post = (
    az.extract(idata_spline, var_names='mu')
    .to_dataframe()
    .reset_index()
)

x_sat_post = (
    az.extract(idata_spline, var_names='x_saturated')
    .to_dataframe()
    .reset_index()
)


mm_alpha_post = (
    az.extract(idata_spline, var_names=['mm_alpha'])
    .to_dataframe()
    .reset_index()
)
mm_lam_post = (
    az.extract(idata_spline, var_names=['mm_lam'])
    .to_dataframe()
    .reset_index()
)

x_adstock_post = (
    az.extract(idata_spline, var_names='x_adstock')
    .to_dataframe()
    .reset_index()
)

post_df = (
    mu_post.merge(x_sat_post, on = ['obs_id', 'chain', 'draw'])
    .merge(mm_alpha_post, on  = ['chain', 'draw', 'channels'])
    .merge(mm_lam_post, on = ['chain', 'draw', 'channels'])
    .merge(x_adstock_post, on = ['obs_id', 'chain', 'draw', 'channels'])
)

post_df['play_caller'] = pd.Categorical(
    [play_callers[i] for i in coach_idx[post_df['obs_id'].values]],
    categories = play_callers
)


post_df_spline = (
    pl.from_pandas(post_df)
    .with_columns(pl.lit('Spline Specification').alias('id')
    )

)


big_data_frame = pl.concat([post_df_hsgp, post_df_spline], how = 'diagonal')

big_data_frame.write_parquet(
    'contributions/mmm-vanilla-beta-personnel-contributions.parquet',
    compression = 'zstd'
)


