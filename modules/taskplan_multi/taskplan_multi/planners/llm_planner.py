import re
import anthropic
from collections import namedtuple

FAILED_COST = 2000

ACTION_COSTS = {
    'pick': 10,
    'place': 10,
    'wash': 30,
    'mix': 10,
    'serve': 10,
    'restock': 30,
    'ask-help': 500,
}

Action = namedtuple('Action', ['name', 'args'])

SYSTEM_PROMPT = """\
You are a task planner for a restaurant with three robots.

=== ROBOTS AND THEIR EXCLUSIVE CAPABILITIES ===
- cook_bot   : move, pick, place, mix
               CANNOT access: servingtable1, servingtable2
- server_bot : move, pick, place, serve, restock
               CANNOT access: stove
- cleaner_bot: move, pick, place, wash
               CANNOT access: stove

=== LOCATIONS ===
Kitchen    : base_cook_bot, stove, countertop, cabinet, sink, fridge, pantry
Serving    : base_server_bot, base_cleaner_bot, servingtable1, servingtable2, bussingcart, shelf

=== ACTION RULES ===
ask-help <robot>
  - Every robot starts INACTIVE. Call this EXACTLY ONCE per robot before its first action.
  - After ask-help the robot stays active for the whole plan. Never call it again for the same robot.

move <robot> <from> <to>
  - Robot must currently be at <from>.

pick <robot> <item> <location>
  - Robot must be at <location>, hand must be free, item must be there.
  - cook_bot cannot pick from servingtable1/servingtable2.
  - server_bot and cleaner_bot cannot pick from stove.

place <robot> <item> <location>
  - Robot must be at <location> and holding <item>.
  - Cannot place at base locations (base_cook_bot, base_server_bot, base_cleaner_bot).

wash <robot> <item>
  - ONLY cleaner_bot can wash.
  - cleaner_bot must be at sink and holding the dirty item.

mix <robot> <food_item> <container> <location>
  - ONLY cook_bot can mix.
  - cook_bot must be at <location>, holding <food_item>, and <container> must be at <location>.
  - <container> must be clean (not dirty). <food_item> must not be empty.

serve <robot> <food_item> <container> <location>
  - ONLY server_bot can serve.
  - server_bot must be at <location> (a servingtable), holding <food_item>.
  - <container> must be at <location> and clean.

restock <robot> <food_item>
  - ONLY server_bot can restock.
  - server_bot must be at pantry and holding the (empty) <food_item>.

=== OUTPUT FORMAT ===
Output one action per line: (action_name arg1 arg2 ...)
Use ONLY object names given in the state below. Never invent names.
No numbering, no explanations, no extra text.\
"""


def _format_state(proc_data, task):
    lines = []

    lines.append("=== ROBOT POSITIONS ===")
    for agent in proc_data.agent_list:
        rob_at = proc_data.restaurant[agent]['rob_at']
        lines.append(f"  {agent}: at {rob_at}, hand free, INACTIVE")

    lines.append("\n=== OBJECT STATES ===")
    for container in proc_data.containers:
        loc = container['assetId']
        children = container.get('children') or []
        for child in children:
            name = child['assetId']
            flags = []
            if child.get('dirty') == 1:
                flags.append('dirty')
            if child.get('empty') == 1:
                flags.append('empty')
            flag_str = f" [{', '.join(flags)}]" if flags else ''
            lines.append(f"  {name}: at {loc}{flag_str}")

    lines.append(f"\n=== GOAL ===\n  {task}")
    return '\n'.join(lines)


class LLMPlanner:
    def __init__(self, model='claude-opus-4-8'):
        self.model = model
        self.client = anthropic.Anthropic()

    def _build_user_prompt(self, proc_data, task):
        state = _format_state(proc_data, task)
        return f"{state}\n\nOutput the plan:"

    def _parse_plan(self, text):
        actions = []
        for line in text.strip().splitlines():
            line = line.strip()
            m = re.match(r'\(\s*(\S+)(.*?)\s*\)$', line)
            if m:
                name = m.group(1)
                args_str = m.group(2).strip()
                args = tuple(args_str.split()) if args_str else ()
                actions.append(Action(name=name, args=args))
        return actions if actions else None

    def _compute_cost(self, plan, proc_data):
        cost = 0
        for action in plan:
            if action.name == 'move':
                src = action.args[1]
                target = action.args[2]
                cost += proc_data.known_cost[src][target]
            else:
                cost += ACTION_COSTS.get(action.name, 0)
        return cost

    def get_cost_and_state_from_task(self, proc_data, task):
        user_prompt = self._build_user_prompt(proc_data, task)
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[{'role': 'user', 'content': user_prompt}],
            )
            text_blocks = [b.text for b in response.content if hasattr(b, 'text')]
            raw_text = '\n'.join(text_blocks)
            print(f"[LLMPlanner] raw response:\n{raw_text}")
            plan = self._parse_plan(raw_text)
        except Exception as e:
            print(f"[LLMPlanner] API error: {e}")
            return None, FAILED_COST

        if not plan:
            return None, FAILED_COST

        cost = self._compute_cost(plan, proc_data)
        return plan, cost

    def get_expected_cost(self, proc_data, task_distribution):
        costs = []
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            costs.append(FAILED_COST if plan is None else cost)
        return sum(costs) / len(costs)
