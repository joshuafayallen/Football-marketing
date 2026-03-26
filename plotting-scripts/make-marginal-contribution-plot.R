library(AllenMisc)
library(MetBrewer)
library(arrow)
library(ggdist)
library(tidyverse)

post_df = read_parquet(
  'contributions/mmm-vanilla-beta-personnel-contributions.parquet'
)


# changed after I plotted it
ordering = rev(c(
  '11 Personnel',
  '12 Personnel',
  '13 Personnel',
  '21 Personnel',
  '22 Personnel'
))

ordering


make_marginals = post_df |>
  mutate(
    marginal_contribution = plogis(mu) - plogis(mu - x_saturated)
  )

marginal_summary = make_marginals |>
  summarise(
    marginal_contribution = mean(marginal_contribution),
    .by = c(play_caller, chain, draw, channels, id)
  ) |>
  mutate(
    numbies = str_extract(channels, '\\d{2}'),
    nice_labels = glue::glue("{numbies} Personnel"),
    nice_labels = as_factor(nice_labels),
    nice_labels = fct_relevel(nice_labels, levels)
  )


small = marginal_summary |>
  filter(play_caller == 'Kyle Shanahan')


ggplot(
  small,
  aes(
    x = marginal_contribution,
    y = nice_labels,
    color = nice_labels,
    fill = nice_labels
  )
) +
  stat_histinterval(
    point_interval = 'median_hdi'
  ) +
  scale_fill_met_d(name = 'Lakota') +
  scale_color_met_d(name = 'Lakota') +
  facet_wrap(vars(id)) +
  labs(x = 'Marginal Contribution', y = NULL, color = NULL, fill = NULL) +
  theme_allen_minimal() +
  theme(legend.position = 'none')

## this plot is honestly fine I just hate that it is on the
## Lets get this onto a real interpretable scale

keep_these = read_parquet(
  'processed-data/processed-dat.parquet'
) |>
  summarise(
    games_called = n(),
    .by = off_play_caller
  ) |>
  filter(
    games_called >= 104
  ) |>
  pull(off_play_caller)

analysis_df = read_parquet('processed-data/processed-dat.parquet') |>
  filter(off_play_caller %in% keep_these)


max_abs = analysis_df |>
  summarise(max(explosive_play_rate)) |>
  pull()

nobs = nrow(analysis_df)
scale_factor = max_abs * nobs / (nobs - 1)

mm_derivative = \(x, alpha, lam) {
  return(
    (alpha * lam) / (lam + x)^2
  )
}

expit_derivative = \(mu) {
  p = plogis(mu)
  return(p * (1 - p))
}

delta_df = post_df |>
  mutate(
    d_expit = expit_derivative(mu = mu),
    d_saturation = mm_derivative(x_adstock, alpha = mm_alpha, lam = mm_lam),
    # 1% increase which would be pretty big
    delta_effect = (d_expit * d_saturation * 0.01) * scale_factor
  )

delta_summary = delta_df |>
  summarise(
    delta_effect = mean(delta_effect),
    .by = c(play_caller, chain, draw, channels, id)
  ) |>
  mutate(
    numbies = str_extract(channels, '\\d{2}'),
    nice_labels = glue::glue("{numbies} Personnel"),
    nice_labels = as_factor(nice_labels),
    nice_labels = fct_relevel(nice_labels, levels)
  )


small = delta_summary |>
  filter(play_caller == 'Kyle Shanahan')


ggplot(
  small,
  aes(
    x = delta_effect,
    y = nice_labels,
    color = nice_labels,
    fill = nice_labels
  )
) +
  stat_histinterval(
    point_interval = 'median_hdi'
  ) +
  scale_fill_met_d(name = 'Lakota') +
  scale_color_met_d(name = 'Lakota') +
  facet_wrap(vars(id)) +
  labs(x = 'Marginal Contribution', y = NULL, color = NULL, fill = NULL) +
  theme_allen_minimal() +
  theme(legend.position = 'none')
