library(ggdist)
library(arrow)
library(MetBrewer)
library(tidyverse)

scheme = schema(
  chain = int64(),
  draw = int64(),
  obs_id = int64(),
  x_saturated_pairs = float64(),
  coach = string(),
  personnel_grouping = string(),
  personnel_contribution = float64(),
  coach_intercept = float64(),
  overall_mean = float64(),
  season = float64(),
  week = float64(),
  tenure_relative = float64()
)


raw_data = open_dataset(
  'contribution',
  format = 'parquet',
  partitioning = c("coach", "personnel_grouping"),
  schema = scheme
) |>
  collect()

raw_data |> glimpse()

make_marginal_lift = raw_data |>
  mutate(
    p_full = plogis(overall_mean),
    p_base = plogis(overall_mean - x_saturated_pairs),
    marginal_lift = p_full - p_base,
    personnel_nice = as.factor(str_remove(personnel_grouping, 'personnel_'))
  ) |>
  filter(
    coach %in% c('Andy Reid', 'Sean McVay', 'Sean Payton', 'Kyle Shanahan')
  )


ggplot(
  make_marginal_lift,
  aes(
    x = marginal_lift,
    y = personnel_nice,
    color = personnel_nice,
    fill = personnel_nice
  )
)
