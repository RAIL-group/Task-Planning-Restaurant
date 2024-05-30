def generate_pddl_problem(domain_name, problem_name, objects, init_states,
                          goal_states):
    """
    Generates a PDDL problem file content.

    :param domain_name: Name of the PDDL domain.
    :param problem_name: Name of the PDDL problem.
    :param objects: Dictionary of objects, where keys are types and values are lists of object names.
    :param init_states: List of strings representing the initial states.
    :param goal_states: List of strings representing the goal states.
    :return: A string representing the content of a PDDL problem file.
    """
    # Start the problem definition
    problem_str = f"(define (problem {problem_name})\n"
    problem_str += f"    (:domain {domain_name})\n"

    # Define objects
    problem_str += "    (:objects\n"
    for obj_type, obj_names in objects.items():
        problem_str += "        " + " ".join(obj_names) + " - " + obj_type + "\n"
    problem_str += "    )\n"

    # Define initial state
    problem_str += "    (:init\n"
    for state in init_states:
        problem_str += "        " + state + "\n"
    problem_str += "    )\n"

    # Define goal state
    problem_str += "    (:goal\n"
    problem_str += "        (and\n"
    for state in goal_states:
        problem_str += "            " + state + "\n"
    problem_str += "        )\n"
    problem_str += "    )\n"

    problem_str += "    (:metric minimize (total-cost))\n"

    # Close the problem definition
    problem_str += ")\n"

    return problem_str
