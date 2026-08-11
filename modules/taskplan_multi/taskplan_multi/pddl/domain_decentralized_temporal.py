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
    Temporal version of domain_decentralized: same single-agent-per-plan,
    no-space-restriction setup, but actions are :durative-action instead of
    instantaneous, solved with a real temporal planner (Temporal Fast
    Downward) instead of classical FF/A*.

    `wait-for` is still here, unlike a first pass at this domain that tried
    to drop it in favor of Timed Initial Literals -- the TFD build actually
    available (neighthan/tfd, a python3 port of TFD v0.4) doesn't support
    :timed-initial-literals or #t (duration-dependent numeric effects) at
    all, so "clear `reserved` at an absolute time" can't be expressed
    without an action. The key difference from the classical domain's
    wait-for: this one is a *fixed-duration durative action with no rob-at
    condition or effect whatsoever*. Because it never touches the robot's
    location or hand, nothing marks it mutex with anything else the same
    robot does, so the temporal search is free to run it fully in the
    background -- starting at t=0, ending exactly at the precomputed
    release time -- while genuinely concurrent, causally-independent
    physical actions (e.g. fetching an unrelated ingredient) happen
    alongside it. `pick`'s `(at start (not (reserved ?obj)))` condition
    still can't be satisfied until wait-for's `(at end ...)` effect fires,
    so ordering relative to the item itself is unaffected -- only the
    world no longer forces everything *else* to stall behind it.

    `(over all (rob-at ?r ?loc))` on every action that keeps the robot in
    place is the mutual-exclusion mechanism: it makes that action's window
    conflict with a concurrent `move` for the same robot (move deletes
    `rob-at` at its own start), which is what stops the temporal search
    from scheduling a single robot to do two things at once. `move` itself
    needs no such guard -- a second action's precondition on the
    destination `rob-at` can't be satisfied until the first move's `(at
    end ...)` effect actually establishes it, so moves are naturally
    serialized by ordinary causal dependency.
    """
    DOMAIN_PDDL_TEMPLATE = """
    (define
    (domain restaurant_decentralized_temporal)

    (:requirements :typing :durative-actions :duration-inequalities
                   :fluents :existential-preconditions)

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
        (release-wait ?obj)   ; precomputed remaining reservation span for
                               ; ?obj at problem-generation time -- this
                               ; solver can't derive it during search
        (total-cost)   ; unused by any action -- exists only because the
                        ; shared problem-generation helper always appends
                        ; a "minimize (total-cost)" metric line
    )

    (:durative-action move
        :parameters (?r - robot ?start - location ?end - location)
        :duration (= ?duration (known-cost ?start ?end))
        :condition (and
            (at start (not (= ?start ?end)))
            (at start (rob-at ?r ?start))
        )
        :effect (and
            (at start (not (rob-at ?r ?start)))
            (at end (rob-at ?r ?end))
        )
    )
    (:durative-action pick
        :parameters (?r - robot ?obj - item ?loc - location)
        :duration (= ?duration 10)
        :condition (and
            (at start (is-at ?obj ?loc))
            (at start (rob-at ?r ?loc))
            (over all (rob-at ?r ?loc))
            (at start (hand-is-free ?r))
            (at start (not (reserved ?obj)))
        )
        :effect (and
            (at start (not (is-at ?obj ?loc)))
            (at start (not (hand-is-free ?r)))
            (at end (is-holding ?r ?obj))
        )
    )
    (:durative-action place
        :parameters (?r - robot ?obj - item ?loc - location)
        :duration (= ?duration 10)
        :condition (and
            (at start (is-holding ?r ?obj))
            (at start (rob-at ?r ?loc))
            (over all (rob-at ?r ?loc))
        )
        :effect (and
            (at start (not (is-holding ?r ?obj)))
            (at end (is-at ?obj ?loc))
            (at end (hand-is-free ?r))
        )
    )
    (:durative-action wash
        :parameters (?r - robot ?i - item)
        :duration (= ?duration 30)
        :condition (and
            (at start (rob-at ?r sink))
            (over all (rob-at ?r sink))
            (at start (is-holding ?r ?i))
            (at start (is-dirty ?i))
        )
        :effect (and
            (at end (not (is-dirty ?i)))
        )
    )
    (:durative-action mix
        :parameters (?r - robot ?i - item ?b - item ?loc - location)
        :duration (= ?duration 10)
        :condition (and
            (at start (type ?r cook_bot))
            (at start (rob-at ?r ?loc))
            (over all (rob-at ?r ?loc))
            (at start (is-at ?b ?loc))
            (at start (not (is-dirty ?b)))
            (at start (not (is-empty ?i)))
            (at start (is-holding ?r ?i))
        )
        :effect (and
            (at end (item-in-item ?i ?b ?loc))
            (at end (is-dirty ?b))
            (at end (is-empty ?i))
        )
    )
    (:durative-action serve
        :parameters (?r - robot ?i - item ?b - item ?loc - location)
        :duration (= ?duration 10)
        :condition (and
            (at start (type ?r server_bot))
            (at start (rob-at ?r ?loc))
            (over all (rob-at ?r ?loc))
            (at start (not (is-empty ?i)))
            (at start (is-holding ?r ?i))
            (at start (not (is-dirty ?b)))
            (at start (is-at ?b ?loc))
        )
        :effect (and
            (at end (meal-served ?i ?b ?loc))
            (at end (is-empty ?i))
            (at end (is-dirty ?b))
        )
    )
    (:durative-action restock
        :parameters (?r - robot ?i - item)
        :duration (= ?duration 30)
        :condition (and
            (at start (rob-at ?r pantry))
            (over all (rob-at ?r pantry))
            (at start (is-holding ?r ?i))
            (at start (is-empty ?i))
        )
        :effect (and
            (at end (not (is-empty ?i)))
        )
    )
    (:durative-action wait-for
        :parameters (?r - robot ?obj - item)
        :duration (= ?duration (release-wait ?obj))
        :condition (at start (reserved ?obj))
        :effect (at end (not (reserved ?obj)))
    )
    )
    """

    types_section = generate_types_section(types_dict)
    domain_pddl = DOMAIN_PDDL_TEMPLATE.format(types_section=types_section)
    return domain_pddl
