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
        "mechanical_depth": {"min": 0.5, "target": 0.9, "weight": 1.0},
        "strategic_depth":  {"min": 0.5, "target": 0.8, "weight": 0.9},
        "reflex_focus":     {"min": 0.5, "target": 1.0, "weight": 1.0},
        "difficulty":       {"min": 0.5, "target": 0.7, "weight": 0.7},
        "systemic_complexity": {"target": 0.6, "weight": 0.5},
    },

    "The Intellectual": {
        "strategic_depth":      {"min": 0.7, "target": 0.9, "weight": 1.0},
        "build_variety":        {"target": 0.8, "weight": 0.8},
        "systemic_complexity":  {"min": 0.7, "target": 1.0, "weight": 1.0},
    },

    "The Casual": {
        "difficulty":          {"max": 0.5, "target": 0.4, "weight": 1.0},
        "punishment":          {"max": 0.5, "target": 0.4, "weight": 0.9},
        "mechanical_depth":    {"target": 0.5, "weight": 0.5},
        "strategic_depth":     {"target": 0.5, "weight": 0.4},
        "replayability":       {"min": 0.6, "target": 0.8, "weight": 0.6},
        "systemic_complexity": {"max": 0.5, "target": 0.3, "weight": 0.8},
    },

    "The Explorer": {
        "narrative_importance": {"min": 0.5, "target": 0.6, "weight": 0.7},
        "player_agency":        {"min": 0.6, "target": 0.7, "weight": 0.8},
        "worldbuilding":        {"min": 0.7, "target": 0.8, "weight": 1.0},
        "openness":             {"min": 0.7, "target": 1.0, "weight": 0.9},
    },

    "The Story Devotee": {
        "narrative_importance": {"min": 0.8, "target": 1.0, "weight": 1.0},
        "drama":                {"min": 0.7, "target": 0.9, "weight": 0.9},
        "worldbuilding":        {"min": 0.7, "target": 0.9, "weight": 0.9},
        "player_agency":        {"target": 0.7, "weight": 0.6},

        # explicit exclusions — Hollow Knight
        "mechanical_depth":     {"max": 0.4, "target": 0.2, "weight": 0.8},
        "reflex_focus":         {"max": 0.3, "target": 0.1, "weight": 0.9},
        "replayability":        {"max": 0.5, "target": 0.3, "weight": 0.4},
    },
}

user_df = pd.DataFrame.from_dict(users, orient="index")

# Load data
df = pd.read_csv("games.csv")

# Separate names and features
game_names = df["name"]
features = df.drop(columns=["name"])
feature_cols = df.columns.drop("name")
X = df[feature_cols].values

# Compute similarity matrix
similarity = cosine_similarity(X, X)

# Wrap in DataFrame for readability
sim_df = pd.DataFrame(similarity, index=game_names, columns=game_names)

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

def most_similar(game_name, top_n=5):
    sims = sim_df.loc[game_name].sort_values(ascending=False)
    return sims.iloc[1:top_n+1]  # skip self

def unpack_user_model(user_model):
    user_pref = {}
    constraints = {}
    weights = {}

    for feature, spec in user_model.items():
        if "target" in spec:
            user_pref[feature] = spec["target"]

        if "min" in spec or "max" in spec:
            constraints[feature] = (
                spec.get("min", None),
                spec.get("max", None),
            )

        if "weight" in spec:
            weights[feature] = spec["weight"]

    return user_pref, constraints, weights

def apply_constraints(games_df, constraints):
    mask = pd.Series(True, index=games_df.index)

    for feature, (min_v, max_v) in constraints.items():
        if feature not in games_df.columns:
            raise ValueError(f"Constraint feature missing: {feature}")

        col = games_df[feature]

        # missing value = violation
        if min_v is not None:
            mask &= col.notna() & (col >= min_v)

        if max_v is not None:
            mask &= col.notna() & (col <= max_v)

    return games_df[mask]

def weighted_masked_similarity(user_pref, game_row, weights):
    num = 0.0
    denom_user = 0.0
    denom_game = 0.0

    for feature, target in user_pref.items():
        if feature not in game_row:
            continue
        if np.isnan(game_row[feature]):
            continue

        w = weights.get(feature, 1.0)
        gv = game_row[feature]

        num += w * target * gv
        denom_user += w * target * target
        denom_game += w * gv * gv

    if denom_user == 0 or denom_game == 0:
        return np.nan

    return num / (np.sqrt(denom_user) * np.sqrt(denom_game))

def recommend_games(
    user_pref,
    game_features,
    game_names,
    constraints=None,
    weights=None,
    top_n=5
):
    df = game_features.copy()

    if constraints:
        df = apply_constraints(df, constraints)

    scores = []

    for name, game_row in df.iterrows():
        score = weighted_masked_similarity(
            user_pref,
            game_row,
            weights or {}
        )
        if not np.isnan(score):
            scores.append((game_names[name], score))

    return (
        pd.DataFrame(scores, columns=["game", "similarity"])
        .set_index("game")
        .sort_values("similarity", ascending=False)
        .head(top_n)
    )

def generate_rec(user):
    user_pref, constraints, weights = unpack_user_model(user)

    return recommend_games(
        user_pref=user_pref,
        game_features=features,
        game_names=game_names,
        constraints=constraints,
        weights=weights,
        top_n=5
    )

# ----- "Main" -----
game = "Guilty Gear Strive"

user_model = users["The Tryhard"]

#plot_space()
#print(most_similar(game))
print(generate_rec(user_model))

