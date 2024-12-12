types_dict = {
    "object": ["robot", "item", "location"],
    "robot": ["cook_bot", "server_bot", "cleaner_bot"],
    "location": [
        "base", "servingtable", "bussingcart", "stove", "cabinet",
        "dishwasher", "countertop", "fridge"
    ],
    "item": [
        "pan", "mug", "bowl", "saltshaker", "sauce",
        "pasta", "oats", "milk", "egg"
    ]
}

def generate_types_section(types_dict):
    """
    Generate the :types section of a PDDL domain dynamically from a dictionary.

    Args:
        types_dict (dict): A dictionary where keys are categories and values are lists of types/objects.

    Returns:
        str: The formatted :types section as a string.
    """
    lines = []
    for base_type, subtypes in types_dict.items():
        if subtypes:
            lines.append(f"{' '.join(subtypes)} - {base_type}")
    return "        " + "\n        ".join(lines)


def get_domain(types_dict=types_dict):
    """
    Generate the PDDL domain with dynamic types based on the provided dictionary.
    
    Args:
        custom_types (dict): A dictionary where the keys are type categories and the values are lists of objects.
                             Example: {"pan": ["pan1", "pan2"], "robot": ["cook_bot", "server_bot"]}
    
    Returns:
        str: The updated PDDL domain as a string.
    """
    # Base PDDL domain template
    DOMAIN_PDDL_TEMPLATE = """
    (define
    (domain restaurant)

    (:requirements :strips :typing :action-costs :existential-preconditions)

    (:types
        {types_section}
    )

    (:predicates
        (rob-at ?r - robot ?loc - location)
        (robot-active ?r - robot)
        (is-at ?obj - item ?loc - location)
        (type ?obj - item ?t - object)
        (hand-is-free ?r - robot)
        (restrict-place-to ?loc - location)
        (is-holding ?r - robot ?obj - item)
        (can-reach ?r - robot ?loc - location)
        (is-dirty ?obj - item)
        (is-cooked ?obj - item)
        (meal-served ?obj - item ?loc - location)
    )

    (:functions
        (known-cost ?start ?end)
        (total-cost)
    )

    (:action move
        :parameters (?r - robot ?start - location ?end - location)
        :precondition (and
            (not (= ?start ?end))
            (rob-at ?r ?start)
            (robot-active ?r)
        )
        :effect (and
            (not (rob-at ?r ?start))
            (rob-at ?r ?end)
            (increase (total-cost) (known-cost ?start ?end))
        )
    )
    (:action pick
        :parameters (?r - robot ?obj - item ?loc - location)
        :precondition (and
            (is-at ?obj ?loc)
            (rob-at ?r ?loc)
            (hand-is-free ?r)
            (can-reach ?r ?loc)
            (robot-active ?r)
        )
        :effect (and
            (not (is-at ?obj ?loc))
            (is-holding ?r ?obj)
            (not (hand-is-free ?r))
            (increase (total-cost) 100)
        )
    )
    (:action place
        :parameters (?r - robot ?obj - item ?loc - location)
        :precondition (and
            (not (hand-is-free ?r))
            (rob-at ?r ?loc)
            (not (restrict-place-to ?loc))
            (is-holding ?r ?obj)
            (can-reach ?r ?loc)
            (robot-active ?r)
        )
        :effect (and
            (is-at ?obj ?loc)
            (not (is-holding ?r ?obj))
            (hand-is-free ?r)
            (increase (total-cost) 100)
        )
    )
    (:action wash
        :parameters (?r - robot ?i - item)
        :precondition (and
            (rob-at ?r dishwasher)
            (is-at ?i dishwasher)
            (is-dirty ?i)
        )
        :effect (and
            (not (is-dirty ?i))
            (increase (total-cost) 100)
        )
    )
    (:action cook
        :parameters (?r - robot ?i - item ?p - item)
        :precondition (and
            (rob-at ?r stove)
            (is-at ?p stove)
            (is-at ?i stove)
            (type ?p pan)
            (not (is-dirty ?p))
        )
        :effect (and
            (is-dirty ?p)
            (is-cooked ?i)
            (increase (total-cost) 100)
        )
    )
    (:action serve
        :parameters (?r - robot ?i - item ?loc - location)
        :precondition (and
            (rob-at ?r ?loc)
            (is-cooked ?i)
            (is-at ?i ?loc)
        )
        :effect (and
            (meal-served ?i ?loc)
            (increase (total-cost) 100)
        )
    )
    )
    """

    # Generate the types section dynamically from the provided dictionary
    types_section = generate_types_section(types_dict)

    # Insert the dynamic types section into the PDDL template
    domain_pddl = DOMAIN_PDDL_TEMPLATE.format(types_section=types_section)
    return domain_pddl
