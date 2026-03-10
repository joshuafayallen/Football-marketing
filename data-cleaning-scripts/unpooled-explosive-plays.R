library(nflreadr)
library(tidyverse)


df = load_participation(
  seasons = c(2017:2025),
  include_pbp = TRUE
) |>
  filter(play_type %in% c('pass', 'run'), season_type == 'REG')


df$play_type |> table()

play_callers = read_csv(
  "https://raw.githubusercontent.com/samhoppen/NFL_public/refs/heads/main/data/all_playcallers.csv"
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
  mutate(play_caller_tenure = season - play_caller_mins)


add_play_callers = df |>
  left_join(
    play_callers,
    join_by(nflverse_game_id == game_id, possession_team == team, season, week)
  )


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
  mutate(
    total_snaps = n(),
    .by = c(off_play_caller, nflverse_game_id, play_type)
  ) |>
  mutate(
    snaps_by_personnel = n(),
    .by = c(
      off_play_caller,
      nflverse_game_id,
      personnel_grouping,
      play_type
    )
  ) |>
  mutate(share = (snaps_by_personnel / total_snaps)) |>
  distinct(
    nflverse_game_id,
    personnel_grouping,
    off_play_caller,
    play_type,
    .keep_all = TRUE
  ) |>
  pivot_wider(
    names_from = personnel_grouping,
    values_from = share,
    values_fill = 0,
    id_cols = c(nflverse_game_id, off_play_caller, play_type)
  )


## okay this looks more like it

make_features = add_play_callers |>
  mutate(
    is_explosive = case_when(
      play_type == 'run' & yards_gained >= 10 ~ 1,
      play_type == 'pass' & yards_gained >= 20 ~ 1,
      .default = 0
    )
  ) |>
  group_by(season, nflverse_game_id, off_play_caller, play_type) |>
  summarise(
    success_rate = mean(success, na.rm = TRUE),
    explosive_play_rate = mean(is_explosive, na.rm = TRUE),
    avg_epa = mean(epa, na.rm = TRUE),
    avg_defenders_in_box = mean(defenders_in_box, na.rm = TRUE)
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
    avg_vegas_wp = mean(vegas_wp),
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
    spread_line,
    total_line,
    avg_vegas_wp, # this is just for the possession team
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

make_passing = add_play_callers |>
  mutate(is_pass = ifelse(play_type == 'pass', 1, 0)) |>
  summarise(
    avg_pass_rate = mean(is_pass),
    avg_cpoe = mean(cpoe, na.rm = TRUE),
    .by = c(nflverse_game_id, off_play_caller)
  )


cleaned_data = make_snap_shares |>
  left_join(make_features) |>
  left_join(make_context_vars) |>
  left_join(make_passing) |>
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
  select(-home_team) |>
  filter_out(is.na(off_play_caller))


arrow::write_parquet(
  cleaned_data,
  'processed-data/explosives-separated.parquet'
)
