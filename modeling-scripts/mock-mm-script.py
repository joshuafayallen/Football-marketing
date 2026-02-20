from pymc_marketing.mmm import NoSaturation
from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
)
from pymc_extras.prior import Prior
import polars as pl
import polars.selectors as cs
import pandas as pd
import numpy as np
import preliz as pz 
import matplotlib.pyplot as plt 

seed = 14993111

RANDOM_SEED = np.random.default_rng(seed = seed)

raw_data = (
    pl.read_parquet('processed-data/processed-dat.parquet')
    .with_columns(
        pl.col('nflverse_game_id').str.extract('(\\d+)').str.to_integer().alias('year'), 
        pl.col('game_date').str.to_datetime()
    ).to_dummies(['surface', 'roof'])
)

raw_splits = raw_data.partition_by(['off_play_caller', 'season'], as_dict=True)


coach_splits = {}

for (coach, year), df in raw_splits.items():
    # Create the clean string key you requested
    key = f"{coach} {year}"
    
    # Filter out personnel groups that have zero variance in THIS specific year
    # This prevents the 'SamplingError' when you run the model later
    
    # Store the dataframe and its unique active channels
    coach_splits[key] = {
        "data": df
    }


niners23 = coach_splits['Kyle Shanahan 2023.0']['data']


niners_clean = niners23.drop(
    ['nflverse_game_id', 'off_play_caller']
)

niners23.select(pl.col("game_date").n_unique())

formation_cols = niners_clean.select(cs.starts_with('personnel')).columns

used_formations = [
    col for col in formation_cols
    if niners_clean.select(pl.col(col).n_unique()).item() > 1 
]
used_formations

formation_cleanup = niners_clean.select(
    pl.all().exclude(
        [c for c in formation_cols if c not in used_formations]
    )
)

total_spend_per_channel = (
    formation_cleanup
    .select(pl.col(used_formations).sum())
)

total = total_spend_per_channel.sum().row(0)[0]

spend_share = total_spend_per_channel.select(
    pl.all() / total
)


prior_sig = (len(used_formations) * spend_share.to_numpy()).flatten()



def logit(p):
    # Clips p to avoid log(0) errors
    p = np.clip(p, 0.001, 0.999)
    return np.log(p / (1 - p))

fig, ax = plt.subplots()
pz.Normal(mu=logit(formation_cleanup['explosive_play_rate'].mean()), sigma = 0.5).plot_pdf()

fig, ax = plt.subplots()

formation_cleanup['explosive_play_rate'].mean()

pz.Beta(alpha = 3, beta = 10).plot_pdf()

3/10

mod_config = {
    'intercept': Prior('Normal',
                        mu = 0, sigma = 2.0), 
    'likelihood': Prior('Beta',
                        alpha = Prior('LogNormal', mu = np.log(3), sigma = 2),
                        beta = Prior('LogNormal',  mu = np.log(10), sigma = 2)),
    'gamma_control': Prior('Normal',
                            mu = 0, 
                            sigma = 1.0), 
    'saturation_beta': Prior('HalfNormal', sigma = 2.0)
}


exclude_these = ['season', 'stadium', 'year', 'success_rate', 'explosive_play_rate']
features = formation_cleanup.drop(exclude_these).to_pandas().reset_index(drop = True)

# Ensure game_date is datetime
features['game_date'] = pd.to_datetime(features['game_date'])

# Get control columns (everything except game_date and formations)
all_control_cols = [c for c in features.columns 
                    if c not in used_formations and c != 'game_date']

y = formation_cleanup['explosive_play_rate'].to_pandas().reset_index(drop = True)
y_clean = y.copy()
if isinstance(y_clean, pd.DataFrame):
    y_clean = y_clean.iloc[:, 0]
y_clean.index = features.index

sampler_kwargs = {
    'progress_bar': True, 
    'nuts_sampler': 'numpyro', 
    'random_seed': RANDOM_SEED
}

mmm = MMM(
    model_config=mod_config,
    date_column='game_date', 
    adstock = GeometricAdstock(l_max = 1), 
    channel_columns = used_formations, 
    saturation = NoSaturation().set_dims_for_all_priors('channels'),
    control_columns= all_control_cols
)


mmm.sample_prior_predictive(features, y)
fig, ax = plt.subplots()
mmm.plot_prior_predictive(ax=ax, original_scale=True)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=4)



mmm.fit(features, y, **sampler_kwargs)


mmm.sample_posterior_predictive(features)


mmm.plot_posterior_predictive()

plt.save


mmm.plot_components_contributions()
mmm.plot_waterfall_components_decomposition()




mmm.model_config

## i think the issue is that we are not neccessarily working with a ton of data and the time dynamics are weird 
## coaches have to 
