from core.recommender import recommend_games
from llm.conversation import GameRecommender

def main():
    recommender = GameRecommender()

    print("=== Game Recommender ===")
    print("Type 'quit' to exit\n")

    # Opening message
    opening = recommender.chat("Hi, I'm looking for a game to play")
    print(f"Recommender: {opening}\n")

    while not recommender.ready:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            return

        reply = recommender.chat(user_input)
        print(f"\nRecommender: {reply}\n")

    print("\n--- Extracting your profile... ---\n")
    profile = recommender.extract_profile()
    print(f"Generated profile:\n{profile}\n")

    print("--- Your top game recommendations ---\n")
    results = recommend_games(user_pref=profile, top_n=3)
    print(results)

if __name__ == "__main__":
    main()