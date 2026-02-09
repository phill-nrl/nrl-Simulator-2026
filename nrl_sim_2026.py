import streamlit as st
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import os
import json
import random

# ================== CONFIG ==================
DEFAULT_SIMULATIONS = 10000

FOLDER = r"C:\Users\phill\Documents\nrl-sim\NRL Data\nrl_2025_data"

MATCHES_FILE = "nrl_2025_match_results.csv"
TEAM_STATS_FILE = "nrl_2025_team_stats.csv"
PLAYER_STATS_FILE = "nrl_2025_player_stats.csv"

LINEUPS_FILE = "saved_lineups.json"

HOME_BOOST = 0.3
FATIGUE_PENALTY = -0.05
BASE_TRY_RATE = 0.018

POSITIONS = [
    "Fullback", "Wing Left", "Centre Left", "Centre Right", "Wing Right",
    "Five-Eighth", "Halfback", "Prop Left", "Hooker", "Prop Right",
    "Second Row Left", "Second Row Right", "Lock",
    "Interchange 1", "Interchange 2", "Interchange 3", "Interchange 4"
]

POSITION_GROUPS = {
    'hitup': ['Prop Left', 'Prop Right', 'Second Row Left', 'Second Row Right', 'Lock', 'Interchange 1', 'Interchange 2', 'Interchange 3', 'Interchange 4'],
    'scoot': ['Hooker', 'Fullback', 'Five-Eighth', 'Halfback'],
    'kick': ['Halfback', 'Five-Eighth', 'Fullback']
}

ROLLING_WINDOW = 10

ATTACK_POSITIONS = ["Fullback", "Wing Left", "Centre Left", "Centre Right", "Wing Right", "Five-Eighth", "Halfback"]
DEFENCE_POSITIONS = ["Prop Left", "Hooker", "Prop Right", "Second Row Left", "Second Row Right", "Lock", "Interchange 1", "Interchange 2", "Interchange 3", "Interchange 4"]

# ================== LOAD DATA ==================
@st.cache_data
def load_data():
    matches = pd.read_csv(MATCHES_FILE)
    team_stats = pd.read_csv(TEAM_STATS_FILE) if os.path.exists(TEAM_STATS_FILE) else None
    players_df = pd.read_csv(PLAYER_STATS_FILE) if os.path.exists(PLAYER_STATS_FILE) else None

    if players_df is not None:
        if 'firstname' not in players_df.columns or 'surname' not in players_df.columns:
            players_df['firstname'] = players_df['displayName'].str.split().str[0]
            players_df['surname'] = players_df['displayName'].str.split().str[-1]

        players_df['display_label'] = players_df.apply(
            lambda row: f"{row['surname']}, {row['firstname'][0]} - #{row.get('jumperNumber', '?')} - {row.get('position', 'UNK')}",
            axis=1
        )

    teams = sorted(team_stats['team_name'].unique()) if team_stats is not None else sorted(set(matches['home_team']) | set(matches['away_team']))

    latest_round = team_stats['round'].max() if team_stats is not None else None

    return matches, team_stats, players_df, teams, latest_round

matches, team_stats, players_df, teams, latest_round = load_data()

# Store players in session state for editing
if 'players' not in st.session_state:
    st.session_state.players = players_df

players = st.session_state.players

# ================== AGGREGATE TEAM RATINGS ==================
@st.cache_data
def compute_ratings(rolling_window=ROLLING_WINDOW):
    team_agg = defaultdict(lambda: {'games': 0, 'tries_scored': 0, 'tries_conceded': 0, 'completion': 0.0, 'errors': 0, 'penalties': 0, 'metres': 0, 'post_contact': 0, 'kicking_metres': 0, 'offloads': 0, 'tackle_efficiency': 0.0})

    grouped = team_stats.groupby('match_id')
    for match_id, group in grouped:
        if len(group) != 2:
            continue

        home_row = group[group['team_location'] == 'home'].iloc[0]
        away_row = group[group['team_location'] == 'away'].iloc[0]

        home_team = home_row['team_name']
        away_team = away_row['team_name']

        team_agg[home_team]['games'] += 1
        team_agg[home_team]['tries_scored'] += home_row['tries']
        team_agg[home_team]['tries_conceded'] += away_row['tries']
        team_agg[home_team]['completion'] += home_row['completionRatePercentage'] / 100
        team_agg[home_team]['errors'] += home_row['handlingErrors']
        team_agg[home_team]['penalties'] += home_row['penaltiesConceded']
        team_agg[home_team]['metres'] += home_row['runMetres']
        team_agg[home_team]['post_contact'] += home_row['postContactMetres']
        team_agg[home_team]['kicking_metres'] += home_row['kickMetres']
        team_agg[home_team]['offloads'] += home_row['offloads']
        tackles = home_row['tackles']
        missed = home_row['missedTackles']
        ineffective = home_row['tacklesIneffective']
        total_tackles = tackles + missed + ineffective
        if total_tackles > 0:
            team_agg[home_team]['tackle_efficiency'] += tackles / total_tackles

        team_agg[away_team]['games'] += 1
        team_agg[away_team]['tries_scored'] += away_row['tries']
        team_agg[away_team]['tries_conceded'] += home_row['tries']
        team_agg[away_team]['completion'] += away_row['completionRatePercentage'] / 100
        team_agg[away_team]['errors'] += away_row['handlingErrors']
        team_agg[away_team]['penalties'] += away_row['penaltiesConceded']
        team_agg[away_team]['metres'] += away_row['runMetres']
        team_agg[away_team]['post_contact'] += away_row['postContactMetres']
        team_agg[away_team]['kicking_metres'] += away_row['kickMetres']
        team_agg[away_team]['offloads'] += away_row['offloads']
        tackles = away_row['tackles']
        missed = away_row['missedTackles']
        ineffective = away_row['tacklesIneffective']
        total_tackles = tackles + missed + ineffective
        if total_tackles > 0:
            team_agg[away_team]['tackle_efficiency'] += tackles / total_tackles

    team_ratings = {}
    for team, s in team_agg.items():
        g = s['games']
        if g == 0:
            continue
        team_ratings[team] = {
            'attack': s['tries_scored'] / g,
            'defence': - (s['tries_conceded'] / g),
            'discipline': - (s['errors'] + s['penalties']) / g,
            'completion': s['completion'] / g,
            'metres': s['metres'] / g,
            'post_contact': s['post_contact'] / g,
            'kicking': s['kicking_metres'] / g,
            'offloads': s['offloads'] / g,
            'tackle_efficiency': s['tackle_efficiency'] / g,
            'games': g
        }

    # Normalize ratings
    keys_to_norm = ['attack', 'defence', 'discipline', 'completion', 'metres', 'post_contact', 'kicking', 'offloads', 'tackle_efficiency']
    for key in keys_to_norm:
        values = [r[key] for r in team_ratings.values() if key in r]
        if not values:
            continue
        mean = np.mean(values)
        std = np.std(values) if len(values) > 1 else 1
        for r in team_ratings.values():
            if key in r:
                r[key] = (r[key] - mean) / std if std > 0 else 0

    return team_ratings

team_ratings = compute_ratings()

# ================== PLAYER TRY SCORER PROBS ==================
def compute_player_probs(rolling_window=ROLLING_WINDOW):
    player_agg = defaultdict(lambda: {'games': 0, 'tries': 0, 'line_breaks': 0, 'tackle_breaks': 0, 'metres': 0, 'post_contact': 0, 'team': '', 'tackle_efficiency': 0.0, 'errors': 0, 'penalties': 0})

    for pid in players['playerId'].unique():
        player_df = players[players['playerId'] == pid].sort_values('round', ascending=False).head(rolling_window)
        g = len(player_df)
        if g == 0:
            continue
        player_agg[pid]['games'] = g
        player_agg[pid]['tries'] = player_df['tries'].sum()
        player_agg[pid]['line_breaks'] = player_df['lineBreaks'].sum()
        player_agg[pid]['tackle_breaks'] = player_df['tackleBreaks'].sum()
        player_agg[pid]['metres'] = player_df['runMetres'].sum()
        player_agg[pid]['post_contact'] = player_df['postContactMetres'].sum()
        player_agg[pid]['team'] = player_df['team_name'].iloc[0]
        total_tackles = player_df['tackles'].sum() + player_df['missedTackles'].sum() + player_df['tacklesIneffective'].sum()
        player_agg[pid]['tackle_efficiency'] = player_df['tackles'].sum() / total_tackles if total_tackles > 0 else 0.0
        player_agg[pid]['errors'] = player_df['handlingErrors'].sum()
        player_agg[pid]['penalties'] = player_df['penaltiesConceded'].sum()

    player_try_probs = {}
    for pid, s in player_agg.items():
        g = s['games']
        if g == 0:
            continue
        avg_tries = s['tries'] / g
        avg_line_breaks = s['line_breaks'] / g
        avg_tackle_breaks = s['tackle_breaks'] / g
        avg_metres = s['metres'] / g
        avg_post_contact = s['post_contact'] / g

        raw_prob = avg_tries * 0.5 + avg_line_breaks * 0.2 + avg_tackle_breaks * 0.2 + (avg_metres / 100) * 0.05 + (avg_post_contact / 50) * 0.05

        player_try_probs[pid] = max(0.001, raw_prob)

    team_player_probs = defaultdict(dict)
    for pid, prob in player_try_probs.items():
        team = player_agg[pid]['team']
        if team:
            team_player_probs[team][pid] = prob

    return team_player_probs, player_agg, player_try_probs

team_player_probs, player_agg, player_try_probs_raw = compute_player_probs()

average_raw_prob = np.mean(list(player_try_probs_raw.values())) if player_try_probs_raw else 0.001

# ================== SIMULATION FUNCTION ==================
class Team:
    def __init__(self, name, ratings=None, lineup=None, exclude=None, player_probs=None, try_scale=1.0):
        default_r = team_ratings.get(name, {'attack': 0, 'defence': 0, 'discipline': 0})
        r = ratings or default_r
        self.name = name
        self.attack = r['attack']
        self.defence = r['defence']
        self.discipline = r['discipline']
        self.score = 0
        self.lineup = lineup or {pos: "None" for pos in POSITIONS}
        self.exclude = exclude or {pos: False for pos in POSITIONS}
        self.active_players = [self.lineup[pos] for pos in POSITIONS if self.lineup[pos] != "None" and not self.exclude[pos]]
        self.player_probs = player_probs or team_player_probs.get(name, {})
        self.try_scale = try_scale

def simulate_match(team1_name, team2_name, home_team_name=None, team1_lineup=None, team1_exclude=None, team2_lineup=None, team2_exclude=None, team1_ratings=None, team2_ratings=None, team1_player_probs=None, team2_player_probs=None, team1_try_scale=1.0, team2_try_scale=1.0, num_sets=80):
    team1 = Team(team1_name, team1_ratings, team1_lineup, team1_exclude, team1_player_probs, team1_try_scale)
    team2 = Team(team2_name, team2_ratings, team2_lineup, team2_exclude, team2_player_probs, team2_try_scale)

    if np.random.random() < 0.5:
        attacking = team1
        defending = team2
    else:
        attacking = team2
        defending = team1

    field_position = 25
    consecutive_def_sets = 0

    tries_by_player = defaultdict(int)

    for _ in range(num_sets):
        tackle_count = 0
        set_over = False
        while tackle_count < 6 and not set_over:
            tackle_count += 1
            home_boost = HOME_BOOST if attacking.name == home_team_name else 0
            field_effect = (field_position - 50) / 50 * 1.5
            fatigue = FATIGUE_PENALTY * max(0, consecutive_def_sets - 2)
            tos = attacking.attack - defending.defence + field_effect + home_boost + fatigue + attacking.discipline * 0.1

            if tackle_count <= 3:
                p_hitup = 0.8
                p_scoot = 0.2
                p_kick = 0.0
            elif tackle_count == 4:
                p_hitup = 0.5
                p_scoot = 0.4
                p_kick = 0.1
            elif tackle_count == 5:
                p_hitup = 0.2
                p_scoot = 0.3
                p_kick = 0.5
            else:
                p_hitup = 0.05
                p_scoot = 0.15
                p_kick = 0.8

            probs_sum = p_hitup + p_scoot + p_kick
            if probs_sum == 0:
                p_hitup = 0.4
                p_scoot = 0.3
                p_kick = 0.3
            else:
                p_hitup /= probs_sum
                p_scoot /= probs_sum
                p_kick /= probs_sum

            action = np.random.choice(['hitup', 'scoot', 'kick'], p=[p_hitup, p_scoot, p_kick])

            if action == 'hitup':
                pos_group = 'hitup'
            elif action == 'scoot':
                pos_group = 'scoot'
            else:
                pos_group = 'kick'

            if attacking.active_players:
                scorer_candidates = []
                for p in attacking.active_players:
                    pos = next((pos for pos, pid in attacking.lineup.items() if pid == p), None)
                    if pos and pos in POSITION_GROUPS[pos_group]:
                        scorer_candidates.append(p)
            else:
                scorer_candidates = list(attacking.player_probs.keys())

            if not scorer_candidates:
                scorer_candidates = list(attacking.player_probs.keys())

            probs = [attacking.player_probs.get(p, average_raw_prob) for p in scorer_candidates]
            total_prob = sum(probs)
            if total_prob > 0:
                probs = [p / total_prob for p in probs]

            try_prob = BASE_TRY_RATE * (1 + 0.3 * tos) * attacking.try_scale
            try_prob = min(max(try_prob, 0.001), 0.12)

            if np.random.random() < try_prob:
                if scorer_candidates:
                    scorer = np.random.choice(scorer_candidates, p=probs)
                    tries_by_player[scorer] += 1
                attacking.score += 4
                if np.random.random() < 0.7:
                    attacking.score += 2
                set_over = True
                consecutive_def_sets = 0
                field_position = 25
                attacking, defending = defending, attacking
            else:
                metres_gain = np.random.normal(8 + tos * 2, 5)
                metres_gain = max(1, min(metres_gain, 100 - field_position))
                field_position += metres_gain

                # Error / Turnover
                if np.random.random() < 0.02:
                    set_over = True
                    consecutive_def_sets = 0
                    field_position = 100 - field_position
                    attacking, defending = defending, attacking
                    continue

                # Penalty
                if np.random.random() < 0.015:
                    field_position = min(100, field_position + 10)

            if not set_over:
                if np.random.random() < 0.05:
                    set_over = True
                    consecutive_def_sets = 0
                    field_position = 100 - field_position
                    attacking, defending = defending, attacking

        if not set_over:
            if np.random.random() < 0.1:
                attacking.score += 1
            consecutive_def_sets += 1
            field_position = 100 - field_position
            attacking, defending = defending, attacking

    s1 = team1.score
    s2 = team2.score
    m = s1 - s2
    t = s1 + s2

    return s1, s2, m, t, dict(tries_by_player)

# ================== ADJUST RATINGS FOR LINEUP ==================
def adjust_ratings_for_lineup(team_name, selected_players, lineup, exclude):
    if not selected_players:
        return team_ratings.get(team_name, {'attack': 0, 'defence': 0, 'discipline': 0})

    attack_players = [p for p in selected_players if next((pos for pos, pid in lineup.items() if pid == p), None) in ATTACK_POSITIONS]
    defence_players = [p for p in selected_players if next((pos for pos, pid in lineup.items() if pid == p), None) in DEFENCE_POSITIONS]

    attack_adjust = sum(player_agg[p]['tries'] / player_agg[p]['games'] for p in attack_players if player_agg[p]['games'] > 0) / len(attack_players) if attack_players else 0
    defence_adjust = -sum(player_agg[p]['tackle_efficiency'] / player_agg[p]['games'] for p in defence_players if player_agg[p]['games'] > 0) / len(defence_players) if defence_players else 0
    discipline_adjust = -sum((player_agg[p]['errors'] + player_agg[p]['penalties']) / player_agg[p]['games'] for p in selected_players if player_agg[p]['games'] > 0) / len(selected_players) if selected_players else 0

    adjusted = {
        'attack': team_ratings[team_name]['attack'] + attack_adjust * 0.3,
        'defence': team_ratings[team_name]['defence'] + defence_adjust * 0.3,
        'discipline': team_ratings[team_name]['discipline'] + discipline_adjust * 0.2
    }

    return adjusted

# ================== AUTO-FILL LINEUP FROM LAST ROUND ==================
def get_last_lineup(team):
    if players is None or players.empty:
        return {pos: "None" for pos in POSITIONS}

    team_df = players[players['team_name'] == team]
    if team_df.empty:
        return {pos: "None" for pos in POSITIONS}

    max_round = team_df['round'].max()
    last_game = team_df[team_df['round'] == max_round].copy()

    if last_game.empty:
        return {pos: "None" for pos in POSITIONS}

    last_game['jumper'] = pd.to_numeric(last_game['jumperNumber'], errors='coerce')
    last_game = last_game.sort_values('jumper')

    lineup = {pos: "None" for pos in POSITIONS}
    used_pids = set()

    def assign(pos_key, keywords, limit=1):
        nonlocal used_pids
        mask = last_game['position'].str.lower().str.contains('|'.join(keywords), na=False)
        mask &= ~last_game['playerId'].isin(used_pids)  # Exclude already used players
        cands = last_game[mask].head(limit)
        for i, (_, row) in enumerate(cands.iterrows()):
            key = pos_key if limit == 1 else f"{pos_key} {i+1}"
            pid = int(row['playerId'])
            lineup[key] = pid
            used_pids.add(pid)

    # Priority order: starters first, then fill gaps
    assign('Fullback',      ['fullback', 'fb'])
    assign('Wing Left',     ['wing', 'w', 'left wing'])
    assign('Wing Right',    ['wing', 'w', 'right wing'])
    assign('Centre Left',   ['centre', 'center', 'ctr', 'left centre'])
    assign('Centre Right',  ['centre', 'center', 'ctr', 'right centre'])
    assign('Five-Eighth',   ['five', '5/8', 'five-eighth', 'stand-off'])
    assign('Halfback',      ['half', 'hb', 'scrum half'])
    assign('Prop Left',     ['prop', 'front row', 'frf', 'left prop'])
    assign('Hooker',        ['hooker', 'hk', 'dummy half'])
    assign('Prop Right',    ['prop', 'front row', 'frf', 'right prop'])
    assign('Second Row Left',['second row', 'sr', 'edge', 'back row', 'left second'])
    assign('Second Row Right',['second row', 'sr', 'edge', 'back row', 'right second'])
    assign('Lock',          ['lock', 'middle forward', 'lf'])

    # Fill interchange with remaining unused players (up to 4)
    remaining = last_game[~last_game['playerId'].isin(used_pids)].head(4)
    for i, (_, row) in enumerate(remaining.iterrows()):
        lineup[f'Interchange {i+1}'] = int(row['playerId'])
        used_pids.add(int(row['playerId']))

    # If any position still "None", fill with any remaining player (rare)
    for pos in POSITIONS:
        if lineup[pos] == "None":
            remaining = last_game[~last_game['playerId'].isin(used_pids)]
            if not remaining.empty:
                pid = int(remaining.iloc[0]['playerId'])
                lineup[pos] = pid
                used_pids.add(pid)

    return lineup

# ================== JSON HELPERS ==================
def convert_to_python(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python(item) for item in obj]
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    else:
        return obj

def safe_load_lineup(d):
    return {k: int(v) if isinstance(v, (float, int, np.number)) and not isinstance(v, bool) else v 
            for k, v in d.items()}

# ================== STREAMLIT APP ==================
st.title("NRL 2025 Simulator")

tab1, tab2, tab3, tab4 = st.tabs(["Match Sim", "Lineup Editor", "Data Update", "Player Management"])

with tab1:
    st.subheader("Match Simulation")
    team1 = st.selectbox("Home Team", teams, index=0)
    team2 = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0)
    simulations = st.number_input("Number of Simulations", min_value=100, max_value=100000, value=DEFAULT_SIMULATIONS)

    if st.button("Run Simulation"):
        wins1 = wins2 = draws = 0
        margins = []
        totals = []
        all_scores = Counter()

        for _ in range(simulations):
            s1, s2, m, t, _ = simulate_match(team1, team2, team1)
            margins.append(m)
            totals.append(t)
            all_scores[(s1, s2)] += 1
            if s1 > s2:
                wins1 += 1
            elif s2 > s1:
                wins2 += 1
            else:
                draws += 1

        st.write(f"**{team1} (home) win %**: {wins1 / simulations * 100:.1f}%")
        st.write(f"**{team2} win %**: {wins2 / simulations * 100:.1f}%")
        st.write(f"**Draw %**: {draws / simulations * 100:.2f}%")
        st.write(f"**Avg margin for {team1}**: {np.mean(margins):.1f} points")
        st.write(f"**Avg total points**: {np.mean(totals):.1f} points")

        st.subheader("Top 10 Most Common Scorelines")
        top_scores = all_scores.most_common(10)
        score_data = []
        for (s1, s2), count in top_scores:
            percentage = (count / simulations) * 100
            winner = team1 if s1 > s2 else team2 if s2 > s1 else "Draw"
            score_data.append({"Scoreline": f"{team1} {s1} - {s2} {team2}", "Winner": winner, "Percentage": f"{percentage:.2f}%"})

        st.table(pd.DataFrame(score_data))

with tab2:
    st.subheader("Lineup Editor")

    # Initialize session state
    if 'lineup_home' not in st.session_state:
        st.session_state.lineup_home = {pos: "None" for pos in POSITIONS}
    if 'lineup_away' not in st.session_state:
        st.session_state.lineup_away = {pos: "None" for pos in POSITIONS}
    if 'exclude_home' not in st.session_state:
        st.session_state.exclude_home = {pos: False for pos in POSITIONS}
    if 'exclude_away' not in st.session_state:
        st.session_state.exclude_away = {pos: False for pos in POSITIONS}

    # Safe auto-load
    if os.path.exists(LINEUPS_FILE):
        try:
            with open(LINEUPS_FILE, 'r') as f:
                lineups = json.load(f)
            st.session_state.lineup_home = safe_load_lineup(lineups.get('home', st.session_state.lineup_home))
            st.session_state.lineup_away = safe_load_lineup(lineups.get('away', st.session_state.lineup_away))
            st.session_state.exclude_home = lineups.get('exclude_home', st.session_state.exclude_home)
            st.session_state.exclude_away = lineups.get('exclude_away', st.session_state.exclude_away)
            st.info("Auto-loaded saved lineups.")
        except Exception as e:
            st.warning(f"Failed to load saved lineups. Using defaults. Error: {e}")

    team1 = st.selectbox("Home Team", teams, index=0, key="home_team_lineup")
    team2 = st.selectbox("Away Team", teams, index=1 if len(teams) > 1 else 0, key="away_team_lineup")

    if players is not None:
        team_players_home = players.sort_values('round', ascending=False).query('team_name == @team1').drop_duplicates('playerId', keep='first')
        team_players_away = players.sort_values('round', ascending=False).query('team_name == @team2').drop_duplicates('playerId', keep='first')

        team_players_list_home = ["None"] + sorted(team_players_home['display_label'].tolist())
        team_players_list_away = ["None"] + sorted(team_players_away['display_label'].tolist())

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{team1} Lineup (Home)**")
            if st.button("Auto-Fill Home from Last Round"):
                st.session_state.lineup_home = get_last_lineup(team1)
                st.session_state.exclude_home = {pos: False for pos in POSITIONS}
                st.success("Home lineup auto-filled from last round!")
                st.rerun()

            for pos in POSITIONS:
                current_pid = st.session_state.lineup_home.get(pos)
                default_index = 0
                if current_pid != "None" and isinstance(current_pid, (int, np.integer)):
                    match = team_players_home[team_players_home['playerId'] == current_pid]
                    if not match.empty:
                        label = match['display_label'].iloc[0]
                        if label in team_players_list_home:
                            default_index = team_players_list_home.index(label)
                selected_label = st.selectbox(
                    f"{pos}",
                    team_players_list_home,
                    index=default_index,
                    key=f"home_{pos}"
                )
                actual_player = "None"
                if selected_label != "None":
                    match = team_players_home[team_players_home['display_label'] == selected_label]
                    if not match.empty:
                        actual_player = int(match['playerId'].iloc[0])
                exclude_home = st.checkbox("Exclude", value=st.session_state.exclude_home.get(pos, False), key=f"exclude_home_{pos}")
                st.session_state.lineup_home[pos] = actual_player
                st.session_state.exclude_home[pos] = exclude_home

        with col2:
            st.write(f"**{team2} Lineup (Away)**")
            if st.button("Auto-Fill Away from Last Round"):
                st.session_state.lineup_away = get_last_lineup(team2)
                st.session_state.exclude_away = {pos: False for pos in POSITIONS}
                st.success("Away lineup auto-filled from last round!")
                st.rerun()

            for pos in POSITIONS:
                current_pid = st.session_state.lineup_away.get(pos)
                default_index = 0
                if current_pid != "None" and isinstance(current_pid, (int, np.integer)):
                    match = team_players_away[team_players_away['playerId'] == current_pid]
                    if not match.empty:
                        label = match['display_label'].iloc[0]
                        if label in team_players_list_away:
                            default_index = team_players_list_away.index(label)
                selected_label = st.selectbox(
                    f"{pos}",
                    team_players_list_away,
                    index=default_index,
                    key=f"away_{pos}"
                )
                actual_player = "None"
                if selected_label != "None":
                    match = team_players_away[team_players_away['display_label'] == selected_label]
                    if not match.empty:
                        actual_player = int(match['playerId'].iloc[0])
                exclude_away = st.checkbox("Exclude", value=st.session_state.exclude_away.get(pos, False), key=f"exclude_away_{pos}")
                st.session_state.lineup_away[pos] = actual_player
                st.session_state.exclude_away[pos] = exclude_away

    with st.expander("Debug: Current Stored Lineup (Home)"):
        st.json({k: v if v == "None" else f"ID {v}" for k, v in st.session_state.lineup_home.items()})

    col_save, col_load, col_reset = st.columns(3)
    with col_save:
        if st.button("Save Lineups"):
            try:
                lineups = {
                    'home': convert_to_python(st.session_state.lineup_home),
                    'away': convert_to_python(st.session_state.lineup_away),
                    'exclude_home': convert_to_python(st.session_state.exclude_home),
                    'exclude_away': convert_to_python(st.session_state.exclude_away)
                }
                with open(LINEUPS_FILE, 'w') as f:
                    json.dump(lineups, f)
                st.success("Lineups saved to JSON!")
            except Exception as e:
                st.error(f"Save failed: {str(e)}")

    with col_load:
        if st.button("Load Saved Lineups"):
            if os.path.exists(LINEUPS_FILE):
                try:
                    with open(LINEUPS_FILE, 'r') as f:
                        lineups = json.load(f)
                    st.session_state.lineup_home = safe_load_lineup(lineups.get('home', {pos: "None" for pos in POSITIONS}))
                    st.session_state.lineup_away = safe_load_lineup(lineups.get('away', {pos: "None" for pos in POSITIONS}))
                    st.session_state.exclude_home = lineups.get('exclude_home', {pos: False for pos in POSITIONS})
                    st.session_state.exclude_away = lineups.get('exclude_away', {pos: False for pos in POSITIONS})
                    st.success("Lineups loaded from JSON!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Load failed: {str(e)}\nTry deleting saved_lineups.json to reset.")
            else:
                st.warning("No saved lineups found.")

    with col_reset:
        if st.button("Reset Lineups"):
            st.session_state.lineup_home = {pos: "None" for pos in POSITIONS}
            st.session_state.lineup_away = {pos: "None" for pos in POSITIONS}
            st.session_state.exclude_home = {pos: False for pos in POSITIONS}
            st.session_state.exclude_away = {pos: False for pos in POSITIONS}
            st.success("Reset complete.")
            st.rerun()

    if st.button("Apply Lineups & Re-run Sim"):
        st.write("Applying lineups and re-running simulation...")

        selected_home = [st.session_state.lineup_home[pos] for pos in POSITIONS if st.session_state.lineup_home[pos] != "None" and not st.session_state.exclude_home[pos]]
        selected_away = [st.session_state.lineup_away[pos] for pos in POSITIONS if st.session_state.lineup_away[pos] != "None" and not st.session_state.exclude_away[pos]]

        adjusted_home = adjust_ratings_for_lineup(team1, selected_home, st.session_state.lineup_home, st.session_state.exclude_home)
        adjusted_away = adjust_ratings_for_lineup(team2, selected_away, st.session_state.lineup_away, st.session_state.exclude_away)

        adjusted_home_probs = {}
        for pos in POSITIONS:
            player = st.session_state.lineup_home[pos]
            if player != "None" and not st.session_state.exclude_home[pos]:
                raw = player_try_probs_raw.get(player, average_raw_prob)
                if "Wing" in pos or "Centre" in pos or pos == "Fullback":
                    pos_factor = 1.25  # Tuned down
                elif "Eighth" in pos or "Halfback" in pos:
                    pos_factor = 1.1  # Tuned down
                elif pos == "Hooker":
                    pos_factor = 0.9
                else:
                    pos_factor = 0.3
                adjusted_home_probs[player] = raw * pos_factor

        adjusted_away_probs = {}
        for pos in POSITIONS:
            player = st.session_state.lineup_away[pos]
            if player != "None" and not st.session_state.exclude_away[pos]:
                raw = player_try_probs_raw.get(player, average_raw_prob)
                if "Wing" in pos or "Centre" in pos or pos == "Fullback":
                    pos_factor = 1.25  # Tuned down
                elif "Eighth" in pos or "Halfback" in pos:
                    pos_factor = 1.1  # Tuned down
                elif pos == "Hooker":
                    pos_factor = 0.9
                else:
                    pos_factor = 0.3
                adjusted_away_probs[player] = raw * pos_factor

        total_home = sum(adjusted_home_probs.values())
        if total_home > 0:
            adjusted_home_probs = {p: prob / total_home for p, prob in adjusted_home_probs.items()}

        total_away = sum(adjusted_away_probs.values())
        if total_away > 0:
            adjusted_away_probs = {p: prob / total_away for p, prob in adjusted_away_probs.items()}

        full_home_raw = sum(player_try_probs_raw[p] for p in team_player_probs.get(team1, {}))
        full_away_raw = sum(player_try_probs_raw[p] for p in team_player_probs.get(team2, {}))
        try_scale_home = min(total_home / full_home_raw if full_home_raw > 0 else 1.0, 1.15)  # Cap for realism
        try_scale_away = min(total_away / full_away_raw if full_away_raw > 0 else 1.0, 1.15)  # Cap for realism

        wins1 = wins2 = draws = 0
        margins = []
        totals = []
        all_scores = Counter()
        all_tries = defaultdict(int)

        for _ in range(DEFAULT_SIMULATIONS):
            s1, s2, m, t, tries = simulate_match(team1, team2, team1, st.session_state.lineup_home, st.session_state.exclude_home, st.session_state.lineup_away, st.session_state.exclude_away, adjusted_home, adjusted_away, adjusted_home_probs, adjusted_away_probs, try_scale_home, try_scale_away)
            margins.append(m)
            totals.append(t)
            all_scores[(s1, s2)] += 1
            if s1 > s2:
                wins1 += 1
            elif s2 > s1:
                wins2 += 1
            else:
                draws += 1

            for player, count in tries.items():
                all_tries[player] += count

        st.subheader("Adjusted Results")
        st.write(f"**{team1} (home) win %**: {wins1 / DEFAULT_SIMULATIONS * 100:.1f}%")
        st.write(f"**{team2} win %**: {wins2 / DEFAULT_SIMULATIONS * 100:.1f}%")
        st.write(f"**Draw %**: {draws / DEFAULT_SIMULATIONS * 100:.2f}%")
        st.write(f"**Avg margin for {team1}**: {np.mean(margins):.1f} points")
        st.write(f"**Avg total points**: {np.mean(totals):.1f}")

        st.subheader("Top 10 Most Common Scorelines")
        top_scores = all_scores.most_common(10)
        score_data = []
        for (s1, s2), count in top_scores:
            percentage = (count / DEFAULT_SIMULATIONS) * 100
            winner = team1 if s1 > s2 else team2 if s2 > s1 else "Draw"
            score_data.append({"Scoreline": f"{team1} {s1} - {s2} {team2}", "Winner": winner, "Percentage": f"{percentage:.2f}%"})

        st.table(pd.DataFrame(score_data))

        st.subheader("Try Scoring Probability for Selected Players")
        home_probs = adjusted_home_probs
        away_probs = adjusted_away_probs

        if not home_probs and not away_probs:
            st.warning("No player probabilities available for the selected lineups.")
        else:
            with st.expander(f"{team1} Selected Players Try Scoring Probabilities"):
                home_data = []
                for player, prob in sorted(home_probs.items(), key=lambda x: x[1], reverse=True):
                    name = players[players['playerId'] == player]['displayName'].iloc[0] if not players[players['playerId'] == player].empty else "Unknown"
                    home_data.append({"Player": name, "Try Probability": f"{prob * 100:.1f}%"})
                st.table(pd.DataFrame(home_data))

            with st.expander(f"{team2} Selected Players Try Scoring Probabilities"):
                away_data = []
                for player, prob in sorted(away_probs.items(), key=lambda x: x[1], reverse=True):
                    name = players[players['playerId'] == player]['displayName'].iloc[0] if not players[players['playerId'] == player].empty else "Unknown"
                    away_data.append({"Player": name, "Try Probability": f"{prob * 100:.1f}%"})
                st.table(pd.DataFrame(away_data))

with tab3:
    st.subheader("Data Update Instructions (R Method)")
    st.text("Follow these steps to update your CSVs with new 2026 games using R (same as 2025):")

    st.text("Step 1: Open RStudio or R console.")

    st.text("Step 2: Install nrlR (run once):")
    st.code('install.packages("nrlR")', language="r")

    st.text("Step 3: Load the library (run every time):")
    st.code('library(nrlR)', language="r")

    st.text("Step 4: Set your working directory (change path if needed, run every time):")
    st.code('setwd(r"C:\\Users\\phill\\Documents\\nrl-sim\\NRL Data\\nrl_2025_data")', language="r")

    st.text("Step 5: Fetch new 2026 match results:")
    st.code('results_2026 <- fetch_results(seasons = 2026, league = "nrl", source = "rugbyleagueproject")', language="r")

    st.text("Step 6: Fetch new 2026 player stats (championdata source):")
    st.code('player_2026 <- fetch_player_stats_championdata(comp = 12755)  # Update comp ID if needed', language="r")

    st.text("Step 7: Fetch new 2026 team stats (championdata source):")
    st.code('team_stats_2026 <- fetch_team_stats_championdata(comp = 12755)', language="r")

    st.text("Step 8: Append match results to existing CSV:")
    st.code('old_matches = read.csv("nrl_2025_match_results.csv")'
            '\ncombined_matches = rbind(old_matches, results_2026)'
            '\nwrite.csv(combined_matches, "nrl_2025_match_results.csv", row.names = FALSE)', language="r")

    st.text("Step 9: Append player stats to existing CSV:")
    st.code('old_players = read.csv("nrl_2025_player_stats.csv")'
            '\ncombined_players = rbind(old_players, player_2026)'
            '\nwrite.csv(combined_players, "nrl_2025_player_stats.csv", row.names = FALSE)', language="r")

    st.text("Step 10: Append team stats to existing CSV:")
    st.code('old_team = read.csv("nrl_2025_team_stats.csv")'
            '\ncombined_team = rbind(old_team, team_stats_2026)'
            '\nwrite.csv(combined_team, "nrl_2025_team_stats.csv", row.names = FALSE)', language="r")

    st.text("Step 11: To find the correct comp ID for 2026 (if 12755 doesn't work):")
    st.code('fetch_cd_comps()', language="r")

    st.text("Step 12: After running all above, refresh this app (Ctrl + Shift + R).")

    st.info("Run steps 3-10 after each NRL round in 2026. Your rolling form will update automatically.")

with tab4:
    st.title("Player & Team Management")

    st.subheader("Fix Player Team (Mid-Season Switches) or Add Rookie")

    manage_team = st.selectbox("Select Team to Edit", [""] + teams, index=0)

    if manage_team and players is not None:
        team_players = players[players['team_name'] == manage_team].drop_duplicates(subset='playerId')

        st.write(f"Current players for {manage_team} ({len(team_players)}):")
        st.dataframe(team_players[['displayName', 'team_name', 'jumperNumber', 'position', 'playerId']])

        st.subheader("Change Team for Existing Player")
        
        # Create nice display options: show label, but keep ID for value
        player_options = [""] + [
            f"{row['display_label']} (ID: {row['playerId']})"
            for _, row in team_players.iterrows()
        ]
        player_choice = st.selectbox("Pick Player", player_options)

        if player_choice and player_choice != "":
            # Extract the playerId from the selected string
            player_to_change_str = player_choice.split("ID: ")[-1].strip(")")
            player_to_change = int(player_to_change_str)
            
            player_row = players[players['playerId'] == player_to_change].iloc[0]
            current_team = player_row['team_name']
            new_team = st.selectbox("New Team", teams, index=teams.index(current_team))
            
            if st.button(f"Change {player_row['displayName']} ({player_row['position']}) to {new_team}"):
                players.loc[players['playerId'] == player_to_change, 'team_name'] = new_team
                st.success(f"{player_row['displayName']} moved to {new_team}!")
                st.rerun()    

    st.subheader("Add New Rookie Player")
    new_name = st.text_input("Display Name (e.g. Lachlan Galvin)")
    new_first = st.text_input("First Name (e.g. Lachlan)")
    new_last = st.text_input("Last Name (e.g. Galvin)")
    new_short = st.text_input("Short Display Name (e.g. Galvin, L)")
    new_team_add = st.selectbox("Team", teams)
    new_position = st.selectbox("Position (for average stats)", POSITIONS + ["Unknown"])
    new_jumper = st.number_input("Jumper Number (optional)", min_value=0, max_value=99, value=0)

    if st.button("Add Rookie"):
        if new_name:
            avg_stats = {}  # Placeholder for average stats
            new_player_id = 1000000 + random.randint(0, 99999)  # Temp unique ID
            new_row = pd.DataFrame({
                'playerId': [new_player_id],
                'displayName': [new_name],
                'firstname': [new_first or new_name.split()[0]],
                'surname': [new_last or new_name.split()[-1]],
                'shortDisplayName': [new_short or (new_last + ', ' + new_first[0] if new_first and new_last else new_name)],
                'team_name': [new_team_add],
                'position': [new_position],
                'jumperNumber': [new_jumper],
                **avg_stats,
                'round': [latest_round or 'Unknown'],
                'match_id': [0],
                'competition_id': [12755],
                'match_status': ['complete'],
                'utc_start': ['Unknown']
            })
            st.session_state.players = pd.concat([players, new_row], ignore_index=True)
            st.success(f"{new_name} added to {new_team_add} with average {new_position} stats! (Assigned temp playerId: {new_player_id})")
            st.rerun()
        else:
            st.warning("Enter a name.")

    st.info("Changes are in-memory (affect lineups immediately).")
    st.info("To save permanently (survives app restart):")
    if st.button("Save All Changes & New Rookies to CSV"):
        st.session_state.players.to_csv(PLAYER_STATS_FILE, index=False)
        st.cache_data.clear()  # Clear cache to reload data
        st.success("Players CSV updated - changes saved!")

        st.rerun()

