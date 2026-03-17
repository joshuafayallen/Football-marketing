library(AllenMisc)
library(arrow)
library(tidybayes)
library(ggdist)
library(ordbetareg)
library(brms)
library(tidyverse)


raw_data = read_parquet('processed-data/processed-dat.parquet')

keep_these = raw_data |>
  group_by(off_play_caller) |>
  summarise(tot = n()) |>
  filter(tot >= 100) |>
  pluck('off_play_caller')


preds = c(
  'avg_epa',
  'avg_defenders_in_box',
  'is_indoors',
  'is_grass',
  'div_game',
  'wind',
  'temp',
  'is_home_team',
  'avg_diff',
  'avg_pass_rate'
)


personnel_means = raw_data |>
  select(starts_with('personnel')) |>
  colMeans()

unreliable_cols = names(personnel_means[personnel_means < 0.01])

cleanish = raw_data |>
  filter(off_play_caller %in% keep_these) |>
  select(-all_of(unreliable_cols)) |>
  mutate(
    is_grass = ifelse(surface == 'grass', TRUE, FALSE),
    is_indoors = ifelse(roof %in% c('closed', 'dome'), TRUE, FALSE),
    across(
      c(starts_with('personnel'), 'explosive_play_rate'),
      \(x) {
        x / max(abs(x))
      },
      .names = '{.col}_scaled'
    ),
    across(
      all_of(preds),
      \(x) {
        (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
      },
      .names = "{.col}_sdz"
    )
  )


## I am going to the geometric adstock function separetly just to check if she works

geometric_adstock = function(spends, alpha, l_max = 12, normalize = FALSE) {
  w = alpha^(0:(l_max - 1))
  if (normalize) {
    w = w / sum(w)
  }

  out = stats::filter(spends, w, sides = 1)

  out[is.na(out)] = sapply(
    which(is.na(out)),
    \(i) sum(w[1:i] * spends[i:1])
  )

  return(as.numeric(out))
}
raw_spend = c(
  1000,
  900,
  800,
  700,
  600,
  500,
  400,
  300,
  200,
  100,
  0,
  0,
  0,
  0,
  0,
  0
)

make_spends =
  tibble(
    t = seq_along(raw_spend),
    `Raw Spend` = raw_spend,
    `Alpha 0.2` = geometric_adstock(
      raw_spend,
      alpha = 0.20,
      l_max = 8,
      normalize = TRUE
    ),
    `Alpha 0.5` = geometric_adstock(
      raw_spend,
      alpha = 0.50,
      l_max = 8,
      normalize = TRUE
    ),
    `Alpha 0.8` = geometric_adstock(
      raw_spend,
      alpha = 0.80,
      l_max = 8,
      normalize = TRUE
    )
  ) |>
  pivot_longer(
    -t,
    names_to = 'alpha_level',
    values_to = 'adspend_values'
  )
