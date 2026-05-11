import json
import os
import re
import difflib
import asyncio
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm

# --- Official SDKs ---
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


# --------------- Path and Environment Settings ---------------

PROVIDER = "gemini"               # TODO: Change to "openai" when using GPT models

MODEL_NAME = "gemini-2.5-flash"   # TODO: Change model name here

DEV_DATA_FILE = "/root/autodl-tmp/FEHU/dataset/dev/dev_event_base.json"  # TODO: Change string here

# Automatically build the directory structure required by the official guidelines
OUTPUT_BASE_DIR = f"/root/autodl-tmp/FEHU/outputs/{MODEL_NAME}"
TASK1_DIR = os.path.join(OUTPUT_BASE_DIR, "task1")
TASK2_DIR = os.path.join(OUTPUT_BASE_DIR, "task2")

# Capture the archive directory from run.py for debug logs
LOG_DIR = os.environ.get("EXP_ARCHIVE_DIR", OUTPUT_BASE_DIR)
FAILED_LOG_FILE = os.path.join(LOG_DIR, "debug_failed_outputs.txt")

# Output paths for all 4 official files
OUTPUT_TASK1A = os.path.join(TASK1_DIR, "pred_task1a.json")
OUTPUT_TASK1B = os.path.join(TASK1_DIR, "pred_task1b.json")
OUTPUT_TASK2A = os.path.join(TASK2_DIR, "pred_task2a.json")
OUTPUT_TASK2B = os.path.join(TASK2_DIR, "pred_task2b.json")

# Paths to all official taxonomy dictionaries
HIERARCHY_FILE = "/root/autodl-tmp/FEHU/dataset/hv_categories/level_2_to_level_1.json"
L1_DICT_FILE = "/root/autodl-tmp/FEHU/dataset/hv_categories/human_value_level1_values.json"
L2_DICT_FILE = "/root/autodl-tmp/FEHU/dataset/hv_categories/human_value_level2_values.json"
DIR_DICT_FILE = "/root/autodl-tmp/FEHU/dataset/hv_categories/direction_mapping.json"

# Thread-safe lists for asyncio results
task1a_results, task1b_results = [], []
task2a_results, task2b_results = [], []

# Metrics for robustness tracking (using dict to safely mutate inside async functions)
metrics = {
    "failed_parse_count": 0,
    "hallucination_count": 0
}

# Concurrency Limiter
CONCURRENCY_LIMIT = 10
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)


# --------------- API Client Adapter ---------------

class LLMClient:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider.lower()
        self.model_name = model_name

        if self.provider == "gemini":
            if not genai: raise ImportError("Please install google-genai SDK.")
            self.api_key = os.environ.get("GEMINI_API_KEY")
            if not self.api_key: raise ValueError("GEMINI_API_KEY env var missing.")
            self.gemini_client = genai.Client(api_key=self.api_key)

        elif self.provider == "openai":
            if not AsyncOpenAI: raise ImportError("Please install openai SDK.")
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key: raise ValueError("OPENAI_API_KEY env var missing.")
            self.openai_client = AsyncOpenAI(api_key=self.api_key)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        if self.provider == "gemini":
            response = await self.gemini_client.aio.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.0,
                    max_output_tokens=3000
                )
            )
            return response.text
        elif self.provider == "openai":
            response = await self.openai_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=3000
            )
            return response.choices[0].message.content.strip()


# --------------- Main Async Processing Logic ---------------

async def process_article(guid: str, article_data: dict, client: LLMClient,
                          l1_id_map: dict, l2_id_map: dict, dir_id_map: dict, system_prompt: str):

    async with semaphore:
        actors = article_data.get("actors", [])
        subevents = article_data.get("subevents", [])

        actor_info = ""
        for actor_dict in actors:
            for name, actor_id in actor_dict.items():
                actor_info += f"- Actor Name: {name} (ID: {actor_id})\n"

        subevents_text = "\n".join([f"[Subevent ID: {se['id']}] {se['subevent']}" for se in subevents])

        user_query = f"""--- News Article ---
{article_data.get('content', '')}

--- Subevents ---
{subevents_text}

--- Actors to Evaluate ---
{actor_info}

Please evaluate the Human Values for the actors above.
Output ONLY a valid JSON object matching the exact structure demonstrated in the [FEW-SHOT EXAMPLE] from the system instructions. Ensure the "guid" is strictly "{guid}"."""

        generated_text = ""  # Pre-initialize to safely catch API connection errors in exception block

        try:
            # 1. Call API
            generated_text = await client.generate_response(system_prompt, user_query)

            # 2. Parse JSON robustly
            match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = generated_text

            prediction = json.loads(json_str)

            # ---------- Extract Task 1 (Article Level) ----------
            article_hv_1a, article_hv_1b = [], []
            for actor_data in prediction.get("article_level_evaluation", []):
                actor_id = actor_data.get("actor")

                # Level 2 (Task 1a)
                l2_vals = actor_data.get("level_2_values", [])
                if isinstance(l2_vals, str): l2_vals = [l2_vals]
                for l2_name in l2_vals:
                    l2_id = l2_id_map.get(l2_name)
                    if l2_id:
                        article_hv_1a.append({"actor": actor_id, "l2_value": l2_id})
                    else:
                        metrics["hallucination_count"] += 1

                # Level 1 (Task 1b)
                l1_objs = actor_data.get("level_1_values", [])
                for l1_obj in l1_objs:
                    if isinstance(l1_obj, dict):
                        l1_name = l1_obj.get("value")
                        dir_name = l1_obj.get("direction", "aligned")
                    else:
                        l1_name = str(l1_obj)
                        dir_name = "aligned"

                    # Attempt Exact Match
                    l1_id = l1_id_map.get(l1_name)

                    # Attempt Fuzzy Match
                    if l1_id is None and l1_name:
                        close_matches = difflib.get_close_matches(l1_name, l1_id_map.keys(), n=1, cutoff=0.85)
                        if close_matches:
                            l1_id = l1_id_map[close_matches[0]]

                    dir_id = dir_id_map.get(dir_name.lower())
                    if l1_id and dir_id is not None:
                        article_hv_1b.append({"actor": actor_id, "direction": dir_id, "l1_value": l1_id})
                    else:
                        metrics["hallucination_count"] += 1

            if article_hv_1a: task1a_results.append({"guid": guid, "article_human_values": article_hv_1a})
            if article_hv_1b: task1b_results.append({"guid": guid, "article_human_values": article_hv_1b})

            # ---------- Extract Task 2 (Subevent Level) ----------
            subevents_hv_2a, subevents_hv_2b = [], []
            for se_data in prediction.get("subevent_level_evaluation", []):
                se_id = str(se_data.get("subevent_id", ""))
                se_hv_2a, se_hv_2b = [], []

                for actor_data in se_data.get("actors", []):
                    actor_id = actor_data.get("actor")

                    # Level 2 (Task 2a)
                    l2_vals = actor_data.get("level_2_values", [])
                    if isinstance(l2_vals, str): l2_vals = [l2_vals]
                    for l2_name in l2_vals:
                        l2_id = l2_id_map.get(l2_name)
                        if l2_id:
                            se_hv_2a.append({"actor": actor_id, "l2_value": l2_id})
                        else:
                            metrics["hallucination_count"] += 1

                    # Level 1 (Task 2b)
                    l1_objs = actor_data.get("level_1_values", [])
                    for l1_obj in l1_objs:
                        if isinstance(l1_obj, dict):
                            l1_name = l1_obj.get("value")
                            dir_name = l1_obj.get("direction", "aligned")
                        else:
                            l1_name = str(l1_obj)
                            dir_name = "aligned"

                        # Attempt Exact Match
                        l1_id = l1_id_map.get(l1_name)

                        # Attempt Fuzzy Match
                        if l1_id is None and l1_name:
                            close_matches = difflib.get_close_matches(l1_name, l1_id_map.keys(), n=1, cutoff=0.85)
                            if close_matches:
                                l1_id = l1_id_map[close_matches[0]]

                        dir_id = dir_id_map.get(dir_name.lower())
                        if l1_id and dir_id is not None:
                            se_hv_2b.append({"actor": actor_id, "direction": dir_id, "l1_value": l1_id})
                        else:
                            metrics["hallucination_count"] += 1

                if se_hv_2a: subevents_hv_2a.append({"subevent_id": se_id, "subevent_human_values": se_hv_2a})
                if se_hv_2b: subevents_hv_2b.append({"subevent_id": se_id, "subevent_human_values": se_hv_2b})

            if subevents_hv_2a: task2a_results.append({"guid": guid, "subevents_human_values": subevents_hv_2a})
            if subevents_hv_2b: task2b_results.append({"guid": guid, "subevents_human_values": subevents_hv_2b})

        except Exception as e:
            metrics["failed_parse_count"] += 1
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(FAILED_LOG_FILE, "a", encoding="utf-8") as df:
                df.write(f"\n--- Failed GUID: {guid} ---\n{generated_text}\nError: {e}\n")


# --------------- Initialization and Main Loop ---------------

async def main():
    os.makedirs(TASK1_DIR, exist_ok=True)
    os.makedirs(TASK2_DIR, exist_ok=True)

    print("[INFO] Loading official dictionaries and hierarchy...")
    with open(HIERARCHY_FILE, 'r', encoding='utf-8') as f: value_hierarchy = json.load(f)["level_2_to_level_1"]
    with open(L1_DICT_FILE, 'r', encoding='utf-8') as f: l1_id_map = json.load(f)
    with open(L2_DICT_FILE, 'r', encoding='utf-8') as f: l2_id_map = json.load(f)
    with open(DIR_DICT_FILE, 'r', encoding='utf-8') as f: dir_id_map = json.load(f)

    hierarchy_text = ""
    for l2_value, l1_list in value_hierarchy.items():
        hierarchy_text += f"- Level 2: {l2_value}\n"
        for l1_value in l1_list:
            hierarchy_text += f"  * Level 1: {l1_value}\n"
    print("[SUCCESS] Dictionaries loaded successfully.")

    system_instruction = f"""You are a highly rigorous sociology and news value analysis expert.
Your task is to read the original news article and evaluate the Human Values demonstrated by the specified Actors at BOTH the Article-level and the Subevent-level.

[ABSOLUTE RULE 1: Official Value Hierarchy (EXACT MATCH REQUIRED)]
Your predicted `level_2_values` and `level_1_values` MUST BE EXACT STRING MATCHES from the provided taxonomy.
Do NOT paraphrase! Do NOT use synonyms! Do NOT invent new values! If it is not in the list below, you score zero.
{hierarchy_text}

[ABSOLUTE RULE 2: Direction for Level 1]
For each Level 1 value, you MUST assign a direction: either "aligned" or "contradictory".

[ABSOLUTE RULE 3: ONE UNIFIED JSON ONLY]
You MUST output EXACTLY ONE complete, unified JSON object containing BOTH the article-level and subevent-level evaluations.
DO NOT split your answer into multiple JSON blocks.
The `reasoning_process` MUST be a string field strictly INSIDE the JSON object. DO NOT write any conversational plaintext outside of the JSON block!

[ABSOLUTE RULE 4: COMPREHENSIVENESS & QUANTITY]
You MUST extract ALL explicitly or implicitly expressed human values.
- For Article-level: Strive to extract AT LEAST 3 to 5 distinct values per actor if the context supports it.
- For Subevent-level: Strive to extract AT LEAST 1 to 3 distinct values per actor.
Do NOT be conservative. If an action implies a value, INCLUDE IT!

[ABSOLUTE RULE 5: FEW-SHOT EXAMPLE (LEARN FROM THIS DENSE & COMPLEX EXAMPLE!)]
Below is a PERFECT example of the EXPECTED JSON STRUCTURE based on a highly complex news event.
Notice how the human expert extracts a HIGH DENSITY of values per actor, successfully identifying BOTH aligned and contradictory directions, and writes a concise "reasoning_process" that accurately justifies these values based on the actor's actions. Do NOT be conservative!

-- EXAMPLE INPUT --
News Article: "White people wanting to visit George Floyd Square, the intersection where the 46-year-old Black man died last May, will need to heed special instructions... The intersection was converted into a memorial and renamed George Floyd Square shortly after his death. A sign near the entrance welcomes visitors to 'A Sacred Space for Community, Public Grief, and Protest.'... However, it includes a special set of instructions for White people. White visitors need to 'decenter' and 'come to listen, learn, mourn, and witness.'... Since Floyd’s death, the area has been converted into an 'autonomous zone' that’s brought record levels of violent crime and gun violence to the neighborhood. Concrete barricades set up to protect demonstrations now act as barriers to an 'autonomous zone' co-opted by armed individuals declaring law enforcement unwelcome..."
Subevents:
[Subevent ID: 0] White visitors to George Floyd Square are given special instructions upon entering the area.
[Subevent ID: 1] A sign welcomes visitors to George Floyd Square, designating it as a sacred space for community, grief, and protest.
[Subevent ID: 2] The sign includes specific guidelines for White people to follow when visiting the Square.
[Subevent ID: 3] Since George Floyd's death, the area has become an autonomous zone with increased violent crime and gun violence.
[Subevent ID: 4] Concrete barricades have been set up, and law enforcement is declared unwelcome in the area.
[Subevent ID: 5] Derek Chauvin was found guilty of multiple charges in George Floyd's death.
[Subevent ID: 6] The guilty verdict sparked relief but highlighted ongoing issues with racial equality in the U.S. justice system.
Actors:
- Actor Name: Community Organizers / Activists at George Floyd Square (ID: 23-12357-4-1-0)
- Actor Name: White Visitors / General Public (ID: 23-12357-4-1-1)
- Actor Name: Armed Autonomy Zone Enforcers (ID: 23-12357-4-1-2)
- Actor Name: Law Enforcement / Police (ID: 23-12357-4-1-3)
- Actor Name: General Community (BIPOC Residents and Supporters) (ID: 23-12357-4-1-4)

-- EXAMPLE OUTPUT --
{{
  "guid": "23-12357-4-1",
  "article_level_evaluation": [
    {{
      "actor": "23-12357-4-1-0",
      "reasoning_process": "The organizers outline specific guidelines for White visitors to ensure they support the BIPOC community respectfully, fostering a more just and equitable environment. They actively foster a sense of belonging by designating the area as a sacred space for community and public grief, which honors these activities as important traditions for social healing. They also emphasize personal accountability and responsibility.",
      "level_1_values": [
        {{"value": "Be just", "direction": "aligned"}},
        {{"value": "Have equality", "direction": "aligned"}},
        {{"value": "Be responsible", "direction": "aligned"}},
        {{"value": "Have a sense of belonging", "direction": "aligned"}},
        {{"value": "Be respecting traditions", "direction": "aligned"}}
      ],
      "level_2_values": [
        "Universalism: concern",
        "Benevolence: dependability",
        "Security: personal",
        "Tradition"
      ]
    }},
    {{
      "actor": "23-12357-4-1-1",
      "reasoning_process": "The instructions for white visitors to 'decenter' and 'come to listen, learn, mourn, and witness' directly call for humility, asking them to set aside their own needs. Furthermore, encouraging visitors to engage in a learning moment rather than escalating conflict promotes a broadminded and tolerant approach.",
      "level_1_values": [
        {{"value": "Be humble", "direction": "aligned"}},
        {{"value": "Be broadminded", "direction": "aligned"}}
      ],
      "level_2_values": [
        "Humility",
        "Universalism: tolerance"
      ]
    }},
    {{
      "actor": "23-12357-4-1-2",
      "reasoning_process": "The transformation of the area into an 'autonomous zone' with armed individuals and gang activity explicitly compromises public safety and undermines social order. This lawlessness creates instability and fear, contradicting the values of a stable society and a safe country.",
      "level_1_values": [
        {{"value": "Have a stable society", "direction": "contradictory"}},
        {{"value": "Have a safe country", "direction": "contradictory"}}
      ],
      "level_2_values": [
        "Security: societal"
      ]
    }},
    {{
      "actor": "23-12357-4-1-3",
      "reasoning_process": "The entire situation is predicated on the actions of former police officer Derek Chauvin, whose killing of George Floyd was deemed unjust by a jury. This initial act of injustice is the catalyst for the unrest.",
      "level_1_values": [
        {{"value": "Be just", "direction": "contradictory"}}
      ],
      "level_2_values": [
        "Universalism: concern"
      ]
    }},
    {{
      "actor": "23-12357-4-1-4",
      "reasoning_process": "The creation of the Square aims to foster a sense of belonging for the BIPOC community by providing a dedicated place where their grief is centered. Additionally, the community views the guilty verdict for Derek Chauvin as a moment of judicial fairness and justice.",
      "level_1_values": [
        {{"value": "Have a sense of belonging", "direction": "aligned"}},
        {{"value": "Be just", "direction": "aligned"}}
      ],
      "level_2_values": [
        "Security: personal",
        "Universalism: concern"
      ]
    }}
  ],
  "subevent_level_evaluation": [
    {{
      "subevent_id": "0",
      "actors": [
        {{
          "actor": "23-12357-4-1-0",
          "reasoning_process": "The special instructions given to White visitors are designed to prevent unintentional harm to the BIPOC community, promoting a more just and equitable form of engagement.",
          "level_1_values": [
            {{"value": "Be just", "direction": "aligned"}}
          ],
          "level_2_values": ["Universalism: concern"]
        }}
      ]
    }},
    {{
      "subevent_id": "1",
      "actors": [
        {{
          "actor": "23-12357-4-1-0",
          "reasoning_process": "By designating the area as a sacred space for community and public grief, the organizers foster a sense of belonging and honor these activities as important traditions.",
          "level_1_values": [
            {{"value": "Have a sense of belonging", "direction": "aligned"}},
            {{"value": "Be respecting traditions", "direction": "aligned"}}
          ],
          "level_2_values": ["Security: personal", "Tradition"]
        }}
      ]
    }},
    {{
      "subevent_id": "2",
      "actors": [
        {{
          "actor": "23-12357-4-1-0",
          "reasoning_process": "The specific directives emphasize the duty for White visitors to support the community responsibly and ensure equitable engagement.",
          "level_1_values": [
            {{"value": "Be responsible", "direction": "aligned"}},
            {{"value": "Have equality", "direction": "aligned"}}
          ],
          "level_2_values": ["Benevolence: dependability", "Universalism: concern"]
        }}
      ]
    }},
    {{
      "subevent_id": "3",
      "actors": [
        {{
          "actor": "23-12357-4-1-2",
          "reasoning_process": "The reported increase in violent crime within the autonomous zone directly disrupts social order, which is the antithesis of a stable and predictable society.",
          "level_1_values": [
            {{"value": "Have a stable society", "direction": "contradictory"}}
          ],
          "level_2_values": ["Security: societal"]
        }}
      ]
    }},
    {{
      "subevent_id": "4",
      "actors": [
        {{
          "actor": "23-12357-4-1-2",
          "reasoning_process": "The erection of barricades and the explicit exclusion of law enforcement directly exacerbate insecurity, undermining the fundamental need for a safe country.",
          "level_1_values": [
            {{"value": "Have a safe country", "direction": "contradictory"}}
          ],
          "level_2_values": ["Security: societal"]
        }}
      ]
    }},
    {{
      "subevent_id": "5",
      "actors": [
        {{
          "actor": "23-12357-4-1-4",
          "reasoning_process": "The guilty verdict for Derek Chauvin reflects legal accountability for an act of police brutality, aligning with the societal value of justice.",
          "level_1_values": [
            {{"value": "Be just", "direction": "aligned"}}
          ],
          "level_2_values": ["Universalism: concern"]
        }}
      ]
    }},
    {{
      "subevent_id": "6",
      "actors": [
        {{
          "actor": "23-12357-4-1-4",
          "reasoning_process": "The public's acknowledgment that the verdict is not the final step underscores a continued pursuit of systemic justice.",
          "level_1_values": [
            {{"value": "Be just", "direction": "aligned"}}
          ],
          "level_2_values": ["Universalism: concern"]
        }}
      ]
    }}
  ]
}}"""

    print(f"\n[INFO] Loading development dataset from: {DEV_DATA_FILE}")
    with open(DEV_DATA_FILE, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    client = LLMClient(provider=PROVIDER, model_name=MODEL_NAME)

    print(f"\n[INFO] Executing batch inference via {PROVIDER.upper()} API on {len(dev_data)} items...")

    tasks = [process_article(item['guid'], item, client, l1_id_map, l2_id_map, dir_id_map, system_instruction) for item in dev_data]

    await tqdm.gather(*tasks, desc="Generating JSONs")

    with open(OUTPUT_TASK1A, 'w', encoding='utf-8') as f: json.dump(task1a_results, f, ensure_ascii=False, indent=4)
    with open(OUTPUT_TASK1B, 'w', encoding='utf-8') as f: json.dump(task1b_results, f, ensure_ascii=False, indent=4)
    with open(OUTPUT_TASK2A, 'w', encoding='utf-8') as f: json.dump(task2a_results, f, ensure_ascii=False, indent=4)
    with open(OUTPUT_TASK2B, 'w', encoding='utf-8') as f: json.dump(task2b_results, f, ensure_ascii=False, indent=4)

    print("\n" + "="*50)
    print("Parsing and Serialization Report")
    print("="*50)
    print(f"Total articles processed: {len(dev_data)}")
    print(f"Failed JSON parses:       {metrics['failed_parse_count']}")
    print(f"Hallucinated labels dropped: {metrics['hallucination_count']}")
    print("-" * 50)
    print(f"[SUCCESS] Task 1 outputs saved to: {TASK1_DIR}/")
    print(f"[SUCCESS] Task 2 outputs saved to: {TASK2_DIR}/")
    print("="*50)

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())