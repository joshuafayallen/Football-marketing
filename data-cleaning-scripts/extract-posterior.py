
import arviz as az
import polars as pl 
import polars.selectors as cs
import pandas as pd 
import numpy as np
import pymc as pm 
import matplotlib.pyplot as plt 
from scipy.special import expit

idata = az.from_netcdf('model/mmm-binomial.nc')




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
    .filter(pl.col('off_play_caller').is_in(keep_these))
    .with_columns(
        ((pl.col('avg_pass_rate') - pl.col('avg_pass_rate').mean())/pl.col("avg_pass_rate").std()).alias('pass_rate_sdz'),
         pl.col('nflverse_game_id')
        .str.extract(r"_(\d{2})_")
        .str.replace_all('_', '')
        .str.to_integer()
        .alias('week')
    )
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

### for now lets just back out the league wide picture 


coach_mean = post['coach_mu'].mean(dim = ('chain', 'draw')).values
control_mean = post['control_contribution'].mean(dim = ('chain', 'draw')).values
passing_mean = (
    post['passing_prior'].mean(dim = ('chain', 'draw')).values *
    raw_data['pass_rate_sdz'].to_numpy()
)



x_sat_mean  = post['x_saturated'].mean(dim=('chain', 'draw')).values  

base_logit = coach_mean + control_mean + passing_mean
base_prob = expit(base_logit)

sort_order = raw_data.to_pandas().sort_values(['off_play_caller', 'season', 'week']).index.values


sorted_coaches = raw_data.to_pandas().values[sort_order]
obs_x  = np.arange(len(sort_order))
obs_exr = raw_data['n_explosive'].to_numpy()/raw_data['total_plays'].to_numpy()

cumulative_logit = base_logit.copy()
prev_prob = base_prob.copy()

fig, ax = plt.subplots(figsize=(16, 6))
colors = plt.cm.tab10.colors

ax.fill_between(obs_x, 0, base_prob[sort_order],
                alpha=0.7, color='gray', label='Base (coach + control)')

coords = {'channels': personnel_scaled.columns}



for i, ch in enumerate(coords['channels']): 
    cumulative_logit = cumulative_logit + x_sat_mean[:, i]
    current_prob = expit(cumulative_logit)
    ax.fill_between(
        obs_x,
        prev_prob[sort_order], 
        current_prob[sort_order], 
        alpha = 0.6,
        color = colors[i],
        label = ch, 

    )

    prev_prob = current_prob


ax.scatter(obs_x, obs_exr[sort_order],
           color='black', alpha=0.2, s=4, label='Observed EPR')

ax.set_xlabel('Game (sorted by coach, season, week)')
ax.set_ylabel('Explosive play rate')
ax.set_ylim(0, 0.18)
ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=8)
ax.set_title('Personnel contribution breakdown — Binomial model')
plt.tight_layout()
plt.show()

coach_mean = post['coach_mu'].mean(dim = ('chain', 'draw')).values
control_mean = post['control_contribution'].mean(dim = ('chain', 'draw')).values
passing_mean = (
    post['passing_prior'].mean(dim = ('chain', 'draw')).values *
    raw_data['pass_rate_sdz'].to_numpy()
)

# lets look at indvidual coaches 

mapping = (
    raw_data 
    .with_row_index("obs_id") 
    .select(["obs_id", "off_play_caller", "season", "week"])
).lazy()

post = pl.from_pandas(az.extract(idata,
                    group = 'posterior',
                    var_names=['passing_contribution',
                                'control_contribution',
                                'coach_mu', 
                                ],
                    combined = True).to_dataframe().reset_index())


x_sat = pl.from_pandas(az.extract(
    idata,
    group='posterior',
    var_names=['x_saturated'],
    combined=True
).to_dataframe().reset_index())


post.write_parquet('contributions/pass-control-mu-post.parquet')
x_sat.write_parquet('contributions/x-sat.parquet')
# restarted in between 
post = pl.scan_parquet('contributions/pass-control-mu-post.parquet')
x_sat = pl.scan_parquet('contributions/x-sat.parquet')



# base: coach identity + controls + passing (no personnel)
