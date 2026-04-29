"""
Command line runner for the Music Recommender.

Usage:
    python src/main.py           # original rule-based recommender (hardcoded profile)
    python src/main.py --agent   # AI agent: preset menu or natural language input
"""

import argparse
from recommender import load_songs, recommend_songs


def main() -> None:
    songs = load_songs("data/songs.csv")

    user_prefs = {
        "name": "Alex",
        "preferred_genre": "pop",
        "preferred_mood": "happy",
        "preferred_energy": 0.75,
        "preferred_valence": 0.68,
        "preferred_danceability": 0.72,
        "preferred_acousticness": 0.28,
        "preferred_tempo_bpm": 118,
        "liked_ids": [10, 18],
        "skipped_ids": [13, 19],
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


def _print_preset_menu(presets: dict) -> None:
    print("\nChoose a vibe:")
    print("─" * 40)
    for key, profile in presets.items():
        print(f"  {key}. {profile['name']}")
        print(f"     {profile['description']}")
    print(f"  {len(presets) + 1}. Describe my own vibe...")
    print("─" * 40)


def _run_refinement(result: dict) -> None:
    """Handle the optional refinement step after initial results are shown."""
    from agent import refine_profile
    from recommender import recommend_songs

    print("\nRefine your results (e.g. 'more upbeat', 'less acoustic', 'faster tempo')")
    refinement = input("Refinement (or press Enter to finish): ").strip()
    if not refinement:
        return

    print("\nRefining...")
    refined_profile = refine_profile(result["profile_used"], refinement)
    pool = [song for song, _, _ in result["results"]]
    pool += [
        s for s in result["candidates"]
        if (s["title"], s["artist"]) not in
        {(r[0]["title"], r[0]["artist"]) for r in result["results"]}
    ]
    final = recommend_songs(refined_profile, pool, k=10)

    print("\nTop 10 refined results:")
    print("─" * 40)
    for i, (song, score, explanation) in enumerate(final, 1):
        print(f"{i:2}. {song['title']} — {song['artist']}")
        print(f"    {song['genre']} | score: {score:.2f}")
        print(f"    {explanation}")
        print()


def _print_results(results: list, label: str, confidence: float, attempts: int) -> None:
    print(f"\nTop 10 for \"{label}\":")
    print(f"Confidence: {confidence:.2f}  |  Attempts: {attempts}")
    print("─" * 40)
    for i, (song, score, _) in enumerate(results[:10], 1):
        print(f"{i:2}. {song['title']} — {song['artist']}")
        print(f"    {song['genre']} | score: {score:.2f}")


def main_agent() -> None:
    from agent import PRESET_PROFILES, run_agent

    print("\nMusic Recommender — AI Agent")
    print("─" * 40)

    # ── Step 1: show menu ──────────────────────────────────────────────────
    _print_preset_menu(PRESET_PROFILES)

    custom_option = str(len(PRESET_PROFILES) + 1)
    choice = input(f"Enter a number (1–{custom_option}): ").strip()

    # ── Step 2: route to preset or natural language flow ──────────────────
    if choice in PRESET_PROFILES:
        selected = PRESET_PROFILES[choice]
        query = selected["name"]           # used as the search query
        preset_profile = selected
        print(f"\nLoading profile: {selected['name']}...")

    elif choice == custom_option:
        query = input("\nDescribe what you want to listen to: ").strip()
        if not query:
            print("No input provided.")
            return
        preset_profile = None
        print("\nSearching catalog...")

    else:
        print(f"Invalid choice. Please enter a number between 1 and {custom_option}.")
        return

    # ── Step 3: run the pipeline ───────────────────────────────────────────
    result = run_agent(query, k=100, preset_profile=preset_profile)

    if "error" in result:
        print(f"\nRejected: {result['error']}")
        return

    label = selected["name"] if choice in PRESET_PROFILES else query
    _print_results(result["results"], label, result["confidence"], result["attempts"])

    # ── Step 4: offer refinement ───────────────────────────────────────────
    _run_refinement(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Use AI agent mode")
    args = parser.parse_args()

    if args.agent:
        main_agent()
    else:
        main()