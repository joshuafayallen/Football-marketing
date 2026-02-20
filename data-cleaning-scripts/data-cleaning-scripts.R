library(nflreadr)
library(tidyverse)
library(MetBrewer)


df = load_participation(
  seasons = c(2017:2025),
  include_pbp = TRUE
) |>
  filter(play_type_nfl %in% c('PASS', 'RUSH'), season_type == 'REG')


play_callers = read_csv(
  "https://raw.githubusercontent.com/samhoppen/NFL_public/2f60ca7a84880f63c4349e5e05d3990a66d13a30/data/all_playcallers.csv"
) |>
  select(
    season,
    team,
    week,
    game_id,
    off_play_caller
  ) |>
  mutate(
    play_caller_mins = min(season),
    .by = off_play_caller
  ) |>
  mutate(play_caller_tenure = season - play_caller_mins) |>
  filter(
    off_play_caller %in%
      c(
        'Kyle Shanahan',
        'Sean McVay',
        'Ben Johnson',
        'Josh McDaniels',
        "Matt LaFleur",
        'Mike McDaniel',
        "Kevin O'Conell",
        'Andy Reid',
        'Liam Coen'
      )
  )


# where this fails is that the play caller data doesn't go through the full season
shanny = play_callers |>
  filter(off_play_caller == 'Kyle Shanahan', season == 2025)


add_play_callers = df |>
  left_join(
    play_callers,
    join_by(nflverse_game_id == game_id, season, week, possession_team == team)
  ) |>
  filter(
    off_play_caller %in%
      c(
        'Kyle Shanahan',
        'Sean McVay',
        'Ben Johnson',
        'Josh McDaniels',
        "Matt LaFleur",
        'Mike McDaniel',
        "Kevin O'Conell",
        'Andy Reid',
        'Liam Coen'
      )
  )

add_play_callers$season |>
  max()

write_csv(add_play_callers, 'raw_data/participation-small.csv')


make_outcomes = add_play_callers |>
  filter(nchar(offense_personnel) > 1) |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    across(c(n_tes, n_rbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs), \(x) str_remove_all(x, "TE|RB| ")),
    personnel_grouping = glue::glue("{n_rbs}{n_tes}_personnel"),
    is_explosive = case_when(
      play_type_nfl == 'RUN' & rushing_yards >= 10 ~ 1,
      play_type_nfl == 'PASS' & receiving_yards >= 20 ~ 1,
      .default = 0
    ),
    offense = ifelse(possession_team == home_team, home_team, away_team),
    defense = ifelse(defteam == home_team, home_team, away_team)
  ) |>
  group_by(season, nflverse_game_id, off_play_caller, play_type_nfl) |>
  mutate(
    success_rate = mean(success, na.rm = TRUE),
    explosive_play_rate = mean(is_explosive, na.rm = TRUE),
    avg_epa = mean(epa, na.rm = TRUE),
    avg_defenders_in_box = mean(defenders_in_box, na.rm = TRUE),
    total_game_snaps = n()
  ) |>

  pivot_wider(
    names_from = personnel_grouping,
    values_from = personnel_grouping,
    values_fn = length,
    values_fill = 0
  ) |>
  summarise(across(everything(), first), .groups = "drop") |>
  select(-c(offense_personnel, defense_personnel)) |>
  mutate(
    across(
      ends_with("_personnel"),
      \(x) x / total_game_snaps,
      .names = "share_{.col}"
    ),
    is_off_home_game = ifelse(offense == home_team, 1, 0)
  )

## lets look at the distribution of offensive formations
## generally I know that kyle uses 21 a ton and sean uses 11 a lot.
## generally I know that kyle uses 21 a ton and sean uses 11 a lot.

check = make_outcomes |>
  filter(season == 2023, off_play_caller == 'Kyle Shanahan')

## so according to summer sports this is impossible since in 2023 season
## the niners used 21 at a 36% clip
sum(check$share_21_personnel)

range(make_outcomes$season)


check_later = add_play_callers |>
  filter(
    season == 2023,
    off_play_caller %in% c('Kyle Shanahan', 'Sean McVay')
  ) |>
  select(
    nflverse_game_id,
    off_play_caller,
    offense_personnel,
    offense_players,
    receiver_id,
    rusher_id
  ) |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    across(c(n_tes, n_rbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs), \(x) str_remove_all(x, "TE|RB| ")),
    personnel_grouping = as.factor(str_glue("{n_rbs}{n_tes}"))
  )

## this looks like Juice is not being counted in this regex so we see an inordinate amount of twelve

check_small = check_later |>
  filter(off_play_caller == 'Kyle Shanahan') |>
  count(personnel_grouping)


juice_gsis = '00-0029892'
## okay so the issue is that the regex is only looking for rbs and tes
## juice is being designated as a FB rather than RB as is convention
check_juice = check_later |>
  filter(
    off_play_caller == 'Kyle Shanahan',
    (receiver_id == juice_gsis | rusher_id == juice_gsis)
  ) |>
  select(
    nflverse_game_id,
    off_play_caller,
    offense_personnel,
    offense_players,
    receiver_id,
    rusher_id,
    nflverse_game_id
  ) |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    n_fbs = str_extract(offense_personnel, "\\d+ FB"),
    across(c(n_tes, n_rbs, n_fbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs, n_fbs), \(x) str_remove_all(x, "TE|RB|FB| ")),
    across(c(n_rbs, n_fbs), \(x) as.numeric(x)),
    n_rbs = as.character(n_rbs + n_fbs),
    personnel_grouping = as.factor(str_glue("{n_rbs}{n_tes}"))
  )

## okay by the eye test this looks a lot better but this
## we ar missing a ton of snaps?
make_snap_shares = add_play_callers |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    n_fbs = str_extract(offense_personnel, "\\d+ FB"),
    across(c(n_tes, n_rbs, n_fbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs, n_fbs), \(x) str_remove_all(x, "TE|RB|FB| ")),
    across(c(n_rbs, n_fbs), \(x) as.numeric(x)),
    n_rbs = as.character(n_rbs + n_fbs),
    personnel_grouping = as.factor(str_glue("personnel_{n_rbs}{n_tes}"))
  ) |>
  mutate(total_snaps = n(), .by = c(off_play_caller, nflverse_game_id)) |>
  mutate(
    snaps_by_personnel = n(),
    .by = c(off_play_caller, nflverse_game_id, personnel_grouping)
  ) |>
  mutate(share = (snaps_by_personnel / total_snaps)) |>
  group_by(personnel_grouping, off_play_caller) |>
  distinct(nflverse_game_id, .keep_all = TRUE) |>
  ungroup() |>
  pivot_wider(
    names_from = personnel_grouping,
    values_from = share,
    values_fill = 0,
    id_cols = c(nflverse_game_id, off_play_caller)
  )


## okay this looks more like it

make_features = add_play_callers |>
  mutate(
    n_tes = str_extract(offense_personnel, "\\d+ TE"),
    n_rbs = str_extract(offense_personnel, "\\d+ RB"),
    across(c(n_tes, n_rbs), \(x) ifelse(is.na(x), 0, x)),
    across(c(n_tes, n_rbs), \(x) str_remove_all(x, "TE|RB| ")),
    personnel_grouping = glue::glue("personnel_{n_rbs}{n_tes}"),
    is_explosive = case_when(
      play_type_nfl == 'RUN' & rushing_yards >= 10 ~ 1,
      play_type_nfl == 'PASS' & receiving_yards >= 20 ~ 1,
      .default = 0
    ),
    is_pass = ifelse(play_type_nfl == 'PASS', 1, 0)
  ) |>
  group_by(season, nflverse_game_id, off_play_caller) |>
  summarise(
    success_rate = mean(success, na.rm = TRUE),
    explosive_play_rate = mean(is_explosive, na.rm = TRUE),
    avg_epa = mean(epa, na.rm = TRUE),
    avg_defenders_in_box = mean(defenders_in_box, na.rm = TRUE),
    avg_pass_rate = mean(is_pass)
  ) |>
  ungroup()


make_context_vars = add_play_callers |>
  mutate(
    offense_score = ifelse(
      possession_team == home_team,
      total_home_score,
      total_away_score
    ),
    defense_score = ifelse(
      defteam == home_team,
      total_home_score,
      total_away_score
    ),
    is_home_team = ifelse(possession_team == home_team, 1, 0),
    diff = offense_score - defense_score,
    avg_diff = mean(diff),
    .by = c(off_play_caller, nflverse_game_id)
  ) |>
  group_by(off_play_caller) |>
  distinct(nflverse_game_id, .keep_all = TRUE) |>
  select(
    nflverse_game_id,
    game_date,
    stadium,
    roof,
    surface,
    div_game,
    home_team,
    wind,
    temp,
    is_home_team,
    avg_diff,
    off_play_caller,
    play_caller_tenure,
  ) |>
  ungroup() |>
  mutate(
    temp = case_when(
      roof %in% c('closed', 'dome') ~ 60,
      is.na(temp) ~ 68,
      .default = temp
    ),
    wind = case_when(
      roof %in% c('closed', 'dome') ~ 0,
      is.na(wind) ~ 8,
      .default = wind
    ),
    surface = str_squish(surface)
  )


cleaned_data = make_snap_shares |>
  left_join(make_features) |>
  left_join(make_context_vars) |>
  mutate(
    surface = case_when(
      nchar(surface) < 1 & stadium == "Levi's® Stadium" ~ 'grass',
      nchar(surface) < 1 & stadium == 'Tottenham Hotspur Stadium' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Estadio Azteca (Mexico City)' ~ 'grass',
      nchar(surface) < 1 & stadium == 'MetLife Stadium' ~ 'fiedldturf',
      nchar(surface) < 1 &
        stadium == 'GEHA Field at Arrowhead Stadium' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Soldier Field' ~ 'grass',
      nchar(surface) < 1 & stadium == "M&T Bank Stadium" ~ 'grass',
      nchar(surface) < 1 & stadium == "Empower Field at Mile High" ~ 'grass',
      nchar(surface) < 1 & stadium == "SoFi Stadium" ~ 'matrixturf',
      # spirtually this is still heinz field
      nchar(surface) < 1 & stadium == 'Acrisure Stadium' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Paycor Stadium' ~ 'a_turf',
      # technically the superdome is now caesars but lets just be sure
      nchar(surface) < 1 &
        stadium == 'Mercedes-Benz Stadium' &
        home_team == 'ATL' ~ 'fieldturf',
      nchar(surface) < 1 & stadium == 'EverBank Stadium' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Highmark Stadium' ~ 'astroturf',
      nchar(surface) < 1 & stadium == 'Gillette Stadium' ~ 'fieldturf',
      nchar(surface) < 1 & stadium == 'AT&T Stadium' ~ 'fieldturf',
      nchar(surface) < 1 & stadium == 'Ford Field' ~ 'fieldturf',
      nchar(surface) < 1 & stadium == 'Arena Corinthians' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Allianz Arena' ~ 'grass',
      nchar(surface) < 1 & stadium == 'Lumen Field' ~ 'fieldturf',
      .default = surface
    ),
    surface = str_squish(surface),
    across(c(surface, roof), \(x) as.factor(x))
  ) |>
  select(-home_team)


arrow::write_parquet(cleaned_data, 'processed-data/processed-dat.parquet')


mean(df$defenders_in_box, na.rm = TRUE)

plot_shares_time = cleaned_data |>
  pivot_longer(
    starts_with('personnel')
  )


check = plot_shares_time |>
  filter(off_play_caller == 'Kyle Shanahan') |>
  mutate(
    week = str_extract(nflverse_game_id, '_\\d{2}_'),
    week = as.factor(str_remove_all(week, '_'))
  ) |>
  filter(value > 0)


ggplot(check, aes(x = week, y = value, fill = name)) +
  geom_col(position = 'dodge') +
  scale_fill_met_d(name = 'Demuth') +
  facet_wrap(vars(season))


check |>
  group_by(name) |>
  summarise(mean(value))


add_play_callers |>
  filter(season == 2025)

check = play_callers |>
  filter(season == 2025)
