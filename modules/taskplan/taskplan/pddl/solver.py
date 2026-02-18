import random
import os
import subprocess
import re

def solve_from_pddl(args, domain_pddl, problem_pddl, heuristic="ff()"):
    DOMAIN = domain_pddl
    PROBLEM = problem_pddl
    rand_id = random.randint(100000, 999999)
    domain_name = f"domain_{rand_id}.pddl"
    problem_name = f"problem_{rand_id}.pddl"

    with open(domain_name, "w") as f:
        f.write(DOMAIN)
    with open(problem_name, "w") as f:
        f.write(PROBLEM)

    # Fast Downward configuration
    fd_script_path = "downward/fast-downward.py"
    search_config = [
        "--search", f"astar({heuristic})"
    ]
    # Example: heuristic="antplan(function=anticipatory_cost_fn)"
    # would produce: --search astar(antplan(function=anticipatory_cost_fn))
    # search_config = [
    #     "--heuristic", "h1=hmax()",
    #     "--heuristic", "h2=antplan(function=anticipatory_cost_fn)",
    #     "--search", "astar(sum([g(), weight(h1,1), weight(h2,1)]))"
    # ]

    command = ["python3", fd_script_path, domain_name, problem_name] + search_config

    try:
        output = subprocess.check_output(
            command, stderr=subprocess.STDOUT, universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[FD Error] Planner failed:\n{e.output}")
        os.remove(domain_name)
        os.remove(problem_name)
        return None, None, None
    finally:
        if os.path.exists(domain_name):
            os.remove(domain_name)
        if os.path.exists(problem_name):
            os.remove(problem_name)

    lines = output.splitlines()
    print(output)
    # log_path = args.log_file  # Or hardcode "/models/antplan_log.txt"

    # with open(log_path, "a") as logf:
    #     capture_block = False
    #     current_block = []

    #     for line in lines:
    #         if "[AntPlan] Current State Heuristic:" in line:
    #             # If a previous block was captured, write it first
    #             if current_block:
    #                 logf.write("\n".join(current_block) + "\n\n")
    #                 current_block = []

    #             capture_block = True
    #             current_block.append(line)
    #             continue

    #         if capture_block:
    #             # If this is a separator line or empty line, close the block
    #             if line.strip() == "" or "----------------------------------------" in line:
    #                 current_block.append(line)
    #                 logf.write("\n".join(current_block) + "\n\n")
    #                 current_block = []
    #                 capture_block = False
    #             else:
    #                 current_block.append(line)

    #     # In case the last block wasn't closed properly
    #     if current_block:
    #         logf.write("\n".join(current_block) + "\n\n")
    plan_cost = None
    plan_length = None

    for line in lines:
        if "Solution found!" in line:
            solution_found = True
        if "Plan cost:" in line:
            plan_cost = int(line.split("Plan cost:")[1].strip())
        if "Plan length:" in line:
            plan_length = int(re.search(r"Plan length:\s*(\d+)", line).group(1))

    # If plan cost is 0 and length is 0 → already at goal, return empty plan
    if plan_cost == 0 and plan_length == 0:
        return [], 0, environment.object_state

    # Parse the actual plan actions from file `sas_plan` (FD writes it automatically)
    plan = []
    plan_file = "sas_plan"
    if os.path.exists(plan_file):
        with open(plan_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(";") or not line:
                    continue
                plan.append(line)
        os.remove(plan_file)
    
    return plan, plan_cost
    