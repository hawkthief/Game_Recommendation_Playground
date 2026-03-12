import os
import json
from pathlib import Path
from google import genai
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path, override=True)
key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=key)

MODEL = "models/gemini-3.1-flash-lite-preview"

# ----- Prompts ----- 

CONVERSATION_SYSTEM_PROMPT = CONVERSATION_SYSTEM_PROMPT = """
You are Alex, a gamer with years of experience across every genre. 
You love talking about games and have strong opinions about them.
A friend has asked you to help them find their next game, and you're genuinely excited to help.

Your goal is to get a feel for what kind of experience they're looking for — not through an interview, 
but through the kind of conversation you'd actually have with a friend about games.

How to behave:
- Talk like a person, not an assistant. Use casual language, show enthusiasm, have opinions.
- React genuinely to what they say. If they mention a game you love, say so. If they say something 
  surprising, react to that too.
- Follow interesting threads. If they say something worth digging into, dig into it before moving on.
- Don't work through a mental checklist. Let the conversation go where it naturally goes.
- Share your own takes. "Oh if you liked that, you might vibe with..." or "Interesting, most people 
  who like X also love Y but some hate it, how do you feel about that?"
- When you bring up games as reference points, use them to probe preferences. 
  "Would you want something more like [game A] or more like [game B]?" is more natural than 
  asking abstract questions.
- If they mention a specific game, use that. Ask what they loved or hated about it. 
  Ask if they want something similar or different. Their relationship with games they know 
  is your best signal.
- Cover ground naturally across: how much they like challenge and mastery, whether story matters 
  to them, how they feel about open worlds vs directed experiences, their tolerance for dark or 
  intense themes, whether they prefer deep one-time experiences or games they can return to.
- Don't ask more than one thing at a time. Ever.
- When you feel like you could describe this person's taste confidently to someone else, 
  end your message with the exact token: [PROFILE_READY] — but only after at least 6 exchanges, 
  and only when you genuinely feel ready, not just because you've covered enough topics.
- Never mention that you are building a profile or that recommendations are coming.
"""

EXTRACTION_PROMPT = """
You are a precise data extractor. Given a conversation between a game recommender and a user,
extract the user's gaming preferences as a JSON profile.

The profile must follow this exact structure — only include dimensions you have clear evidence for:
{{
    "dimension_name": {{
        "target": <float 0.0 to 1.0>,
        "weight": <float 0.0 to 1.0>,
        "mode": <"symmetric" | "at_least" | "at_most">
    }},
    ...
}}

Available dimensions:
- mechanical_depth: how much the game rewards mastering complex mechanics
- strategic_depth: how much the game rewards planning and decision-making
- reflex_focus: how much the game demands fast reactions
- build_variety: how many viable ways to approach the game exist
- narrative_importance: how central story is to the experience
- narrative_complexity: how layered and complex the storytelling is
- player_agency: how much player choices affect the world/story
- worldbuilding: how rich and detailed the game world is
- drama: how emotionally intense the story beats are
- darkness: how dark, disturbing or heavy the themes are
- comedy: how much humor is present
- difficulty: how hard the game is overall
- hostility: how threatening and dangerous the game environment feels
- setback: how punishing failure is
- grindiness: how much repetitive effort is required
- progression_speed: how fast the player gets stronger or advances
- openness: how open and non-linear the game world is
- replayability: how much value the game has after the first playthrough
- loop_centrality: how much the game revolves around a core repeated loop
- systemic_complexity: how many interlocking systems the game has

Mode guide:
- at_least: player wants this dimension to be high (e.g. wants rich worldbuilding)
- at_most: player wants this dimension to be low (e.g. wants low difficulty)
- symmetric: player wants a specific value, not just high or low

Conversation to extract from:
{conversation}

Respond with ONLY valid JSON. No explanation, no markdown, no backticks.
"""

# ----- Conversation -----

class GameRecommender:
    def __init__(self):
        self.history = []
        self.ready = False

    def _build_prompt(self, user_message):
        self.history.append({
            "role": "user",
            "content": user_message
        })

        full_prompt = CONVERSATION_SYSTEM_PROMPT + "\n\n"
        for turn in self.history:
            if turn["role"] == "user":
                full_prompt += f"User: {turn['content']}\n"
            else:
                full_prompt += f"You: {turn['content']}\n"
        full_prompt += "You:"

        return full_prompt

    def chat(self, user_message):
        prompt = self._build_prompt(user_message)

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        reply = response.text.strip()

        if "[PROFILE_READY]" in reply:
            self.ready = True
            reply = reply.replace("[PROFILE_READY]", "").strip()

        self.history.append({
            "role": "assistant",
            "content": reply
        })

        return reply

    def extract_profile(self):
        conversation_text = ""
        for turn in self.history:
            role = "User" if turn["role"] == "user" else "Recommender"
            conversation_text += f"{role}: {turn['content']}\n"

        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )

        raw = response.text.strip()
        return json.loads(raw)