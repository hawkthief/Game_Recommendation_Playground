import pandas as pd
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ----- Users -----
users = {
    
    # ----- Archetype users -----
    "The Tryhard": {
        "mechanical_depth": {"min": 0.5, "target": 0.9, "weight": 1.0, "mode": "at_least"},
        "strategic_depth":  {"min": 0.5, "target": 0.8, "weight": 0.8, "mode": "at_least"},
        "reflex_focus":     {"min": 0.5, "target": 1.0, "weight": 1.0, "mode": "symmetric"},
        "difficulty":       {"min": 0.4, "target": 0.7, "weight": 0.8, "mode": "symmetric"},
        "systemic_complexity": {"target": 0.6, "weight": 0.6, "mode": "symmetric"},
    },

    "The Intellectual": {
        "strategic_depth":      {"min": 0.5, "target": 0.9, "weight": 1.0, "mode": "at_least"},
        "build_variety":        {"target": 0.7, "weight": 0.8, "mode": "at_least"},
        "systemic_complexity":  {"min": 0.3, "target": 1.0, "weight": 1.0, "mode": "symmetric"},
        "narrative_complexity":  {"min": 0.5, "target": 0.8, "weight": 0.6, "mode": "symmetric"},
        "reflex_focus":     {"max": 0.6, "target": 0.2, "weight": 0.8, "mode": "symmetric"},
    },

    "The Casual": {
        "difficulty":          {"max": 0.5, "target": 0.4, "weight": 0.8, "mode": "at_most"},
        "setback":          {"max": 0.5, "target": 0.3, "weight": 1.0, "mode": "at_most"},
        "mechanical_depth":    {"target": 0.5, "weight": 0.5, "mode": "at_most"},
        "strategic_depth":     {"target": 0.5, "weight": 0.4, "mode": "at_most"},
        "replayability":       {"min": 0.3, "target": 0.8, "weight": 0.4, "mode": "at_least"},
        "systemic_complexity": {"max": 0.5, "target": 0.3, "weight": 0.8, "mode": "at_most"},
    },

    "The Explorer": {
        "narrative_importance": {"min": 0.4, "target": 0.6, "weight": 0.6, "mode": "at_least"},
        "player_agency":        {"min": 0.6, "target": 0.7, "weight": 0.8, "mode": "at_least"},
        "worldbuilding":        {"min": 0.5, "target": 0.8, "weight": 1.0, "mode": "at_least"},
        "openness":             {"min": 0.7, "target": 1.0, "weight": 0.9, "mode": "at_least"},
    },

    "The Story Devotee": {
        "narrative_importance": {"min": 0.8, "target": 1.0, "weight": 1.0, "mode": "at_least"},
        "drama":                {"min": 0.7, "target": 0.9, "weight": 0.9, "mode": "at_least"},
        "worldbuilding":        {"min": 0.7, "target": 0.9, "weight": 0.9, "mode": "at_least"},
        "player_agency":        {"target": 0.7, "weight": 0.6, "mode": "at_least"},

        # explicit exclusions
        "replayability":        {"max": 0.7, "target": 0.3, "weight": 0.4, "mode": "at_most"},
    },

    "The Horror Fan": {
        "darkness": {"min": 0.6, "target": 0.8, "weight": 1.0, "mode": "at_least"},
        "hostility":                {"min": 0.7, "target": 0.9, "weight": 0.9, "mode": "at_least"},
    },
}

user_df = pd.DataFrame.from_dict(users, orient="index")

# Load data
df = pd.read_csv("games.csv", delimiter=";")

# Separate names and features
game_names = df["name"]
features = df.drop(columns=["name"])
feature_cols = df.columns.drop("name")
X = df[feature_cols].values

# Compute similarity matrix
similarity = cosine_similarity(X, X)

# Wrap in DataFrame for readability
sim_df = pd.DataFrame(similarity, index=game_names, columns=game_names)

def most_similar(game_name, top_n=5):
    sims = sim_df.loc[game_name].sort_values(ascending=False)
    return sims.iloc[1:top_n+1]  # skip self

def penalized_distance(user_spec, game_row, epsilon=0.05, alpha=3.0):
    total = 0.0

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        g = game_row[f]
        t = rules["target"]
        w = rules["weight"]
        mode = rules.get("mode", "symmetric")

        delta = directional_delta(g, t, mode)
        severity = violation_severity(delta, epsilon)
        w_eff = inflated_weight(w, severity, alpha)

        total += w_eff * severity**2

    return total

def l1_distance(user_spec, game_row):
    total = 0.0

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        total += rules["weight"] * abs(game_row[f] - rules["target"])

    return total

def l2_distance(user_spec, game_row):
    total = 0.0

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        total += rules["weight"] * (game_row[f] - rules["target"]) ** 2

    return np.sqrt(total)

def penalized_contributions(user_spec, game_row, epsilon=0.05):
    contribs = {}

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        t = rules["target"]
        g = game_row[f]
        w = rules["weight"]

        delta = abs(g - t)
        if delta > epsilon:
            contribs[f] = w * (delta - epsilon) ** 2
        else:
            contribs[f] = 0.0

    return contribs

def l1_contributions(user_spec, game_row):
    contribs = {}

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        contribs[f] = rules["weight"] * abs(game_row[f] - rules["target"])

    return contribs


def l2_contributions(user_spec, game_row):
    contribs = {}

    for f, rules in user_spec.items():
        if "target" not in rules or "weight" not in rules:
            continue
        if f not in game_row or np.isnan(game_row[f]):
            continue

        contribs[f] = rules["weight"] * (game_row[f] - rules["target"]) ** 2

    return contribs

def recommend_games(
    user_pref,
    game_features,
    game_names,
    constraints=None,
    weights=None,
    top_n=5
):
    games_df = game_features.copy()
    user_axes = user_pref.keys()
    rows = []

    for idx, row in games_df.iterrows():
        if violates_hard_constraints(user_pref, row):
            continue
        
        rows.append({
            "game": game_names[idx],
            "penalized": penalized_distance(user_pref, row),
            "l1": l1_distance(user_pref, row),
            "l2": l2_distance(user_pref, row),
        })

    res = pd.DataFrame(rows).set_index("game").sort_values(by="penalized", ascending=False).head(top_n)
    return res

def violates_hard_constraints(user_spec, game_row):
    for f, rules in user_spec.items():
        g = game_row.get(f, np.nan)

        if "min" in rules and g < rules["min"]:
            return True
        if "max" in rules and g > rules["max"]:
            return True

    return False

def directional_delta(g, t, mode):
    if mode == "symmetric":
        return abs(g - t)
    elif mode == "at_most":
        return max(0, t - g)
    elif mode == "at_least":
        return max(0, g - t)

def violation_severity(delta, epsilon):
    return max(0, delta - epsilon)

def inflated_weight(w, severity, alpha=3.0):
    return w * np.exp(alpha * severity)

def plot_space():
    # Important: standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # PCA to 2 dimensions
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)


    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1])

    for i, name in enumerate(game_names):
        plt.text(X_pca[i, 0] + 0.01, X_pca[i, 1] + 0.01, name, fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Game Feature Space (PCA)")

    plt.show()

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def plot_disagreement():
    scores = recommend_games(user_pref=user_model, game_features=features, game_names=game_names, top_n=5)
    scores_norm = normalize(scores)
    # ----- "Metric vs Metric" -----
    plt.figure(figsize=(6,6))
    plt.scatter(scores_norm["l1"], scores_norm["l2"])

    for game in scores_norm.index:
        plt.text(
            scores_norm.loc[game, "l1"] + 0.01,
            scores_norm.loc[game, "l2"] + 0.01,
            game,
            fontsize=8
        )

    plt.xlabel("L1 distance")
    plt.ylabel("L2 distance")
    plt.title("Metric disagreement: L1 vs L2")
    plt.show()

    # ----- "PCA vs Metric" -----

    from sklearn.decomposition import PCA

    X = scores_norm[["penalized", "l1", "l2"]].values
    X_pca = PCA(n_components=2).fit_transform(X)

    plt.figure(figsize=(6,6))
    plt.scatter(X_pca[:,0], X_pca[:,1])

    for i, game in enumerate(scores_norm.index):
        plt.text(X_pca[i,0], X_pca[i,1], game, fontsize=8)

    plt.title("Metric Sensitivity Space")
    plt.show()

def plot_feature_contributions(game_name, user_model, top_k=8):
    idx = game_names[game_names == game_name].index[0]
    row = features.loc[idx]

    pen = pd.Series(penalized_contributions(user_model, row))
    l1  = pd.Series(l1_contributions(user_model, row))
    l2  = pd.Series(l2_contributions(user_model, row))

    df = pd.DataFrame({
        "penalized": pen,
        "l1": l1,
        "l2": l2
    }).fillna(0)

    # Keep only relevant features
    df["total"] = df.sum(axis=1)
    df = df.sort_values("total", ascending=False).head(top_k)
    df = df.drop(columns="total")

    if df.sum().sum() == 0:
        print("No significant contributions for this game.")
        return

    df.plot(kind="bar", figsize=(9, 4))
    plt.title(f"Feature contribution by metric — {game_name}")
    plt.ylabel("Distance contribution")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# ----- "Main" -----
game = "Fifa 26"
user_model = users["The Horror Fan"]

# Space Analysis
#plot_space()
#print(most_similar(game))

# Metric Analysis
#plot_disagreement()
#plot_feature_contributions(game, user_model)

# Recommendation
print(recommend_games(user_pref=user_model, game_features=features, game_names=game_names, top_n=5))


