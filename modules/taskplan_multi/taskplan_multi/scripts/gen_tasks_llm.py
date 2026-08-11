"""
Generate 15 restaurant tasks (5 per robot) using an LLM, scoped to the
current domain's items, locations, and actions.

The LLM is given only the domain facts and asked to freely propose tasks —
no task-primitive library is provided, so the output reveals what kinds of
goals the model naturally thinks of.

Usage:
    python gen_tasks_llm.py [--model claude-sonnet-4-6] [--output tasks_out.json]
"""

import argparse
import json
import anthropic

# ── Domain facts ────────────────────────────────────────────────────────────

ITEMS = {
    "washable_containers": ["mug1", "mug2", "pan1", "pan2", "bowl1", "bowl2"],
    "food_items":          ["pasta", "oats", "cereal", "milk"],
    "condiments":          ["sauce", "saltshaker"],
}

LOCATIONS = {
    "kitchen":      ["countertop", "cabinet", "stove", "fridge", "pantry", "sink"],
    "serving_room": ["servingtable1", "servingtable2", "bussingcart", "shelf"],
}

ROBOTS = {
    "cook_bot": {
        "actions":    ["move", "pick", "place", "mix"],
        "restricted": ["servingtable1", "servingtable2"],
    },
    "server_bot": {
        "actions":    ["move", "pick", "place", "serve", "restock"],
        "restricted": ["stove"],
    },
    "cleaner_bot": {
        "actions":    ["move", "pick", "place", "wash"],
        "restricted": ["stove"],
    },
}

ACTION_DESCRIPTIONS = """\
- move(from, to)                  : robot moves between two locations
- pick(item, location)            : robot picks up an item at its current location (hand must be free)
- place(item, location)           : robot places the held item at its current location
- mix(food, container, location)  : cook_bot combines a food item into a container at a location
- serve(food, container, location): server_bot presents food in a container to a guest at a serving table
- restock(food_item)              : server_bot refills an empty food item from the pantry
- wash(container)                 : cleaner_bot cleans a dirty container at the sink"""

SYSTEM_PROMPT = """\
You are a task designer for a multi-robot restaurant planning system.
Your job is to propose realistic, varied tasks that each robot could receive during restaurant service.
Each task is defined by a goal state — what should be true when the task is complete.
Respond only with valid JSON — no markdown fences, no extra text."""

USER_PROMPT_TEMPLATE = """\
=== RESTAURANT DOMAIN ===

Items in the restaurant:
  Washable containers : {washable}
  Food items          : {food}
  Condiments          : {condiments}

Locations:
  Kitchen             : {kitchen}
  Serving room        : {serving}

Robots and their capabilities:
  cook_bot
    actions   : {cook_actions}
    CANNOT access : {cook_restrict}

  server_bot
    actions   : {server_actions}
    CANNOT access : {server_restrict}

  cleaner_bot
    actions   : {cleaner_actions}
    CANNOT access : {cleaner_restrict}

Action semantics:
{action_desc}

=== YOUR JOB ===

Generate exactly 15 tasks — 5 for each robot.

Rules:
  1. Each task must be achievable by that robot using only its listed actions and accessible locations.
  2. Use only items and locations named above.
  3. All 15 tasks must be distinct.
  4. Be realistic for a restaurant setting.
  5. Describe the goal state in plain English (what is true when the task is done).
  6. Do NOT reference any specific programming API or function names.

Return a JSON object with this exact schema:
{{
  "cook_bot": [
    {{
      "name": "<short_snake_case_name>",
      "description": "<one sentence describing what the robot must do>",
      "goal": "<what must be true when the task is complete>"
    }},
    ...
  ],
  "server_bot":  [ ... ],
  "cleaner_bot": [ ... ]
}}
"""


def build_prompt() -> str:
    return USER_PROMPT_TEMPLATE.format(
        washable=", ".join(ITEMS["washable_containers"]),
        food=", ".join(ITEMS["food_items"]),
        condiments=", ".join(ITEMS["condiments"]),
        kitchen=", ".join(LOCATIONS["kitchen"]),
        serving=", ".join(LOCATIONS["serving_room"]),
        cook_actions=", ".join(ROBOTS["cook_bot"]["actions"]),
        cook_restrict=", ".join(ROBOTS["cook_bot"]["restricted"]),
        server_actions=", ".join(ROBOTS["server_bot"]["actions"]),
        server_restrict=", ".join(ROBOTS["server_bot"]["restricted"]),
        cleaner_actions=", ".join(ROBOTS["cleaner_bot"]["actions"]),
        cleaner_restrict=", ".join(ROBOTS["cleaner_bot"]["restricted"]),
        action_desc=ACTION_DESCRIPTIONS,
    )


def query_llm(model: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_prompt()}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def print_summary(tasks: dict):
    for robot, task_list in tasks.items():
        print(f"\n{'─' * 56}")
        print(f"  {robot}  ({len(task_list)} tasks)")
        print(f"{'─' * 56}")
        for i, t in enumerate(task_list, 1):
            print(f"  {i}. [{t['name']}]")
            print(f"     What : {t['description']}")
            print(f"     Goal : {t['goal']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate restaurant tasks via LLM (no primitive constraints)"
    )
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Anthropic model ID")
    parser.add_argument("--output", default=None,
                        help="Write raw JSON output to this file")
    args = parser.parse_args()

    print(f"Querying {args.model} for 15 tasks …\n")
    tasks = query_llm(args.model)

    print_summary(tasks)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(tasks, f, indent=2)
        print(f"\nJSON written to: {args.output}")


if __name__ == "__main__":
    main()
