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
        "mechanical_depth": 0.9,
        "strategic_depth": 0.8,
        "reflex_focus": 1.0,
        "difficulty": 0.7,
        "systemic_complexity": 0.6,
    },

    "The Intellectual": {
        "strategic_depth": 0.9,
        "build_variety": 0.8,
        "systemic_complexity": 1.0,
    },

    "The Casual": {
        "mechanical_depth": 0.5,
        "strategic_depth": 0.5,
        "difficulty": 0.4,
        "punishment": 0.4,
        "replayability": 0.8,
        "systemic_complexity": 0.3,
    },

    "The Explorer": {
        "narrative_importance": 0.6,
        "player_agency": 0.7,
        "worldbuilding": 0.8,
        "openness": 1.0,
    },

    "The Completionist": {
        "replayability": 0.9,
        "build_variety": 0.8,
        "openness": 0.7,
        "player_agency": 0.7,
        "systemic_complexity": 0.6,
        "grindiness": 0.6,
        "progression_speed": 0.4,
        "narrative_importance": 0.5,
    },

    "The Story Devotee": {
        "narrative_importance": 1.0,
        "drama": 0.9,
        "worldbuilding": 0.9,
        "player_agency": 0.7,
        "mechanical_depth": 0.2,
        "reflex_focus": 0.1,
        "replayability": 0.3,
    },

    "The Systems Engineer": {
        "systemic_complexity": 1.0,
        "strategic_depth": 0.9,
        "build_variety": 0.8,
        "mechanical_depth": 0.6,
        "narrative_importance": 0.2,
        "drama": 0.1,
    },

    "The Adrenaline Seeker": {
        "reflex_focus": 1.0,
        "difficulty": 0.6,
        "punishment": 0.5,
        "mechanical_depth": 0.6,
        "narrative_importance": 0.2,
        "replayability": 0.7,
    },

    "The Comfort Gamer": {
        "difficulty": 0.2,
        "punishment": 0.2,
        "comedy": 0.6,
        "narrative_importance": 0.5,
        "progression_speed": 0.6,
        "systemic_complexity": 0.3,
    },

    # ----- Dense / semi-random users -----
    "Dense Hybrid 1": {
        "mechanical_depth": 0.7,
        "strategic_depth": 0.6,
        "reflex_focus": 0.5,
        "narrative_importance": 0.6,
        "drama": 0.7,
        "darkness": 0.4,
        "punishment": 0.5,
        "replayability": 0.6,
        "systemic_complexity": 0.6,
    },

    "Dense Hybrid 2": {
        "strategic_depth": 0.8,
        "build_variety": 0.7,
        "openness": 0.6,
        "player_agency": 0.8,
        "worldbuilding": 0.7,
        "mechanical_depth": 0.4,
        "difficulty": 0.5,
    },

    "Dense Hybrid 3": {
        "reflex_focus": 0.8,
        "difficulty": 0.7,
        "punishment": 0.6,
        "mechanical_depth": 0.7,
        "narrative_importance": 0.3,
        "comedy": 0.2,
        "replayability": 0.8,
    },

    "Dense Hybrid 4": {
        "narrative_importance": 0.7,
        "drama": 0.8,
        "darkness": 0.6,
        "player_agency": 0.5,
        "systemic_complexity": 0.4,
        "mechanical_depth": 0.4,
        "difficulty": 0.4,
    },

    "Dense Hybrid 5": {
        "mechanical_depth": 0.3,
        "strategic_depth": 0.9,
        "reflex_focus": 0.2,
        "systemic_complexity": 0.8,
        "narrative_importance": 0.6,
        "comedy": 0.5,
        "punishment": 0.4,
        "openness": 0.5,
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

def most_similar(game_name, top_n=5):
    sims = sim_df.loc[game_name].sort_values(ascending=False)
    return sims.iloc[1:top_n+1]  # skip self

def masked_similarity(user_vec, game_vec):
    # mask where user has expressed a preference
    mask = ~np.isnan(user_vec)

    if mask.sum() == 0:
        return np.nan  # user has no preferences at all

    u = user_vec[mask]
    g = game_vec[mask]

    if norm(u) == 0 or norm(g) == 0:
        return 0.0

    return np.dot(u, g) / (norm(u) * norm(g))

def dominant_axis_filter(user_vec, game_df, top_k_axes=2, percentile=0.6):
    # user_vec is a pandas Series
    expressed = user_vec.dropna()

    if expressed.empty:
        return game_df

    # pick top-k user axes
    dominant_axes = expressed.sort_values(ascending=False).head(top_k_axes).index

    # keep games strong in at least one dominant axis
    mask = pd.Series(False, index=game_df.index)

    for axis in dominant_axes:
        if axis in game_df.columns:
            threshold = game_df[axis].quantile(percentile)
            mask |= game_df[axis] >= threshold

    return game_df[mask]

def recommend_games_for_user(user_name, user_df, game_df, top_n=5):
    user_vec = user_df.loc[user_name].values

    scores = {}
    for game in game_df.index:
        game_name = game_df.loc[game].values[0]  # Extract the game name from the first element
        game_vec = game_df.loc[game].values[1:]  # Skip the first element (game name)
        score = masked_similarity(user_vec, game_vec)
        scores[game_name] = score

    return (
        pd.Series(scores)
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )

def recommend_games_two_step(
    user_name,
    user_df,
    game_df,
    top_n=5,
    top_k_axes=2,
    percentile=0.6
):
    user_vec = user_df.loc[user_name]

    # Step 1: focus
    filtered_games = dominant_axis_filter(
        user_vec,
        game_df,
        top_k_axes=top_k_axes,
        percentile=percentile
    )

    # Step 2: rank
    scores = {}
    for game in filtered_games.index:
        game_name = game_df.loc[game].values[0]  # Extract the game name from the first element
        game_vec = game_df.loc[game].values[1:]
        score = masked_similarity(user_vec, game_vec)
        scores[game_name] = score

    return (
        pd.Series(scores)
        .dropna()
        .sort_values(ascending=False)
        .head(top_n)
    )

# ----- "Main" -----
game = "Guilty Gear Strive"
gamer = "The Tryhard"

#plt.show()
#print(most_similar(game))
#print(title+"\n",recommend_games_for_user(gamer, user_df, df))
#print(title+"\n",recommend_games_two_step(gamer, user_df, df))