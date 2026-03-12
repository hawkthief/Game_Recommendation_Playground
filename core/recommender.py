import pandas as pd
import numpy as np

Epsilon = 0.02
Alpha = 1.5

# ----- Users -----
users = {
    "The Tryhard": {
        "mechanical_depth": {"target": 0.9, "weight": 1.0, "mode": "at_least"},
        "strategic_depth":  {"target": 0.8, "weight": 0.8, "mode": "at_least"},
        "reflex_focus":     {"target": 1.0, "weight": 1.0, "mode": "symmetric"},
        "difficulty":       {"target": 0.7, "weight": 0.8, "mode": "symmetric"},
        "systemic_complexity": {"target": 0.6, "weight": 0.6, "mode": "symmetric"},
    },
    "The Intellectual": {
        "strategic_depth":     {"target": 0.9, "weight": 1.0, "mode": "at_least"},
        "systemic_complexity": {"target": 1.0, "weight": 1.0, "mode": "symmetric"},
        "narrative_complexity":{"target": 0.8, "weight": 0.4, "mode": "symmetric"},
    },
    "The Casual": {
        "difficulty":          {"target": 0.4, "weight": 0.8, "mode": "at_most"},
        "setback":             {"target": 0.3, "weight": 1.0, "mode": "at_most"},
        "mechanical_depth":    {"target": 0.5, "weight": 0.5, "mode": "at_most"},
        "strategic_depth":     {"target": 0.5, "weight": 0.4, "mode": "at_most"},
        "replayability":       {"target": 0.8, "weight": 0.4, "mode": "at_least"},
        "systemic_complexity": {"target": 0.3, "weight": 0.8, "mode": "at_most"},
    },
    "The Explorer": {
        "narrative_importance": {"target": 0.6, "weight": 0.6, "mode": "at_least"},
        "player_agency":        {"target": 0.7, "weight": 0.8, "mode": "at_least"},
        "worldbuilding":        {"target": 0.8, "weight": 1.0, "mode": "at_least"},
        "openness":             {"target": 1.0, "weight": 0.9, "mode": "at_least"},
    },
    "The Story Devotee": {
        "narrative_importance": {"target": 1.0, "weight": 1.0, "mode": "at_least"},
        "drama":                {"target": 0.9, "weight": 0.9, "mode": "at_least"},
        "worldbuilding":        {"target": 0.9, "weight": 0.9, "mode": "at_least"},
        "player_agency":        {"target": 0.7, "weight": 0.6, "mode": "at_least"},
        "replayability":        {"target": 0.3, "weight": 0.4, "mode": "at_most"},
    },
    "The Horror Fan": {
        "darkness":  {"target": 0.8, "weight": 1.0, "mode": "at_least"},
        "hostility": {"target": 0.9, "weight": 0.9, "mode": "at_least"},
    },
}

# ----- Load data -----
df = pd.read_csv("data/games.csv", delimiter=";")
game_names = df["name"]
features = df.drop(columns=["name"])

# ----- Scoring -----
def directional_delta(g, t, mode):
    if mode == "symmetric":
        return abs(g - t)
    elif mode == "at_most":
        return max(0, g - t)
    elif mode == "at_least":
        return max(0, t - g)

def violation_severity(delta, epsilon):
    return max(0, delta - epsilon)

def inflated_weight(w, severity, alpha=Alpha):
    return w * np.exp(alpha * severity)

def penalized_distance(user_spec, game_row, epsilon=Epsilon):
    total = 0.0
    for f, rules in user_spec.items():
        if f not in game_row or np.isnan(game_row[f]):
            continue
        g = game_row[f]
        t = rules["target"]
        w = rules["weight"]
        mode = rules.get("mode", "symmetric")
        delta = directional_delta(g, t, mode)
        severity = violation_severity(delta, epsilon)
        total += inflated_weight(w, severity) * severity**2
    return total

# ----- Recommendation -----
def recommend_games(user_pref, top_n=5):
    rows = []
    for idx, row in features.iterrows():
        rows.append({
            "game": game_names[idx],
            "score": penalized_distance(user_pref, row),
        })
    return (pd.DataFrame(rows)
              .set_index("game")
              .sort_values(by="score", ascending=True)
              .head(top_n))
 
