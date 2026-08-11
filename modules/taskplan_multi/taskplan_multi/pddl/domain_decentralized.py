DEFAULT_TYPES = {
    "object": ["robot", "item", "location"],
    "robot": ["cook_bot", "server_bot"],
    "location": [
        "base", "servingtable", "bussingcart", "stove", "cabinet",
        "sink", "countertop", "fridge", "pantry", "shelf"
    ],
    "item": [
        "pan", "mug", "bowl", "saltshaker", "sauce",
        "pasta", "oats", "milk", "cereal"
    ]
}


def generate_types_section(types_dict):
    lines = []
    for base_type, subtypes in types_dict.items():
        if subtypes:
            lines.append(f"{' '.join(subtypes)} - {base_type}")
    return "        " + "\n        ".join(lines)


def get_domain(types_dict=DEFAULT_TYPES):
    """
    Decentralized, single-agent-per-plan restaurant domain.

    Unlike taskplan_multi.pddl.domain (which gates every action behind a
    shared `robot-active` flag so one classical plan can borrow a second
    robot's actions via `ask-help`), this domain never puts more than one
    robot's actions in a plan — each robot only ever appears as the sole
    `robot` object in its own problem. There is also no
    restrict-reach/restrict-place-to ("no restriction on spaces").

    Cross-robot contention is handled entirely through `reserved`: an item
    another robot has already broadcast a plan for can't be `pick`ed until
    this robot explicitly executes `wait-for` on it — an ordinary action,
    not a special-cased delay.
    """
    DOMAIN_PDDL_TEMPLATE = """
    (define
    (domain restaurant_decentralized)

    (:requirements :strips :typing :action-costs :existential-preconditions)

    (:types
        {types_section}
    )

    (:predicates
        (rob-at ?r - robot ?loc - location)
        (is-at ?obj - item ?loc - location)
        (type ?obj - item ?t - object)
        (hand-is-free ?r - robot)
        (is-holding ?r - robot ?obj - item)
        (is-dirty ?obj - item)
        (is-empty ?obj - item)
        (reserved ?obj - item)
        (meal-served ?obj1 - item ?obj2 - item ?loc - location)
        (item-in-item ?obj1 - item ?obj2 - item ?loc - location)
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
        )
        :effect (and
            (not (rob-at ?r ?start))
            (rob-at ?r ?end)
            ;(increase (total-cost) (known-cost ?start ?end))
        )
    )
    (:action pick
        :parameters (?r - robot ?obj - item ?loc - location)
        :precondition (and
            (is-at ?obj ?loc)
            (rob-at ?r ?loc)
            (hand-is-free ?r)
            (not (reserved ?obj))
        )
        :effect (and
            (not (is-at ?obj ?loc))
            (is-holding ?r ?obj)
            (not (hand-is-free ?r))
            (increase (total-cost) 10)
        )
    )
    (:action place
        :parameters (?r - robot ?obj - item ?loc - location)
        :precondition (and
            (not (hand-is-free ?r))
            (rob-at ?r ?loc)
            (is-holding ?r ?obj)
        )
        :effect (and
            (is-at ?obj ?loc)
            (not (is-holding ?r ?obj))
            (hand-is-free ?r)
            (increase (total-cost) 10)
        )
    )
    (:action wash
        :parameters (?r - robot ?i - item)
        :precondition (and
            ;(type ?r cleaner_bot)
            (rob-at ?r sink)
            (is-holding ?r ?i)
            (is-dirty ?i)
        )
        :effect (and
            (not (is-dirty ?i))
            (increase (total-cost) 30)
        )
    )
    (:action mix
        :parameters (?r - robot ?i - item ?b - item ?loc - location)
        :precondition (and
            (type ?r cook_bot)
            (rob-at ?r ?loc)
            (is-at ?b ?loc)
            (not (is-dirty ?b))
            (not (is-empty ?i))
            (is-holding ?r ?i)
        )
        :effect (and
            (item-in-item ?i ?b ?loc)
            (is-dirty ?b)
            (is-empty ?i)
            (increase (total-cost) 10)
        )
    )
    (:action serve
        :parameters (?r - robot ?i - item ?b - item ?loc - location)
        :precondition (and
            (type ?r server_bot)
            (rob-at ?r ?loc)
            (not (is-empty ?i))
            (is-holding ?r ?i)
            (not (is-dirty ?b))
            (is-at ?b ?loc)
        )
        :effect (and
            (meal-served ?i ?b ?loc)
            (is-empty ?i)
            (is-dirty ?b)
            (increase (total-cost) 10)
        )
    )
    (:action restock
        :parameters (?r - robot ?i - item)
        :precondition (and
            ;(type ?r server_bot)
            (rob-at ?r pantry)
            (is-holding ?r ?i)
            (is-empty ?i)
        )
        :effect (and
            (not (is-empty ?i))
            (increase (total-cost) 30)
        )
    )
    (:action wait-for
        :parameters (?r - robot ?obj - item)
        :precondition (and
            (reserved ?obj)
        )
        :effect (and
            (not (reserved ?obj))
            (increase (total-cost) 1)
        )
    )
    )
    """

    types_section = generate_types_section(types_dict)
    domain_pddl = DOMAIN_PDDL_TEMPLATE.format(types_section=types_section)
    return domain_pddl
