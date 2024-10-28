def get_domain():
    DOMAIN_PDDL = """
    (define
    (domain restaurant)

    (:requirements :strips :typing :action-costs :existential-preconditions)

    (:types
        robot location item - object
        agent_tall agent_tiny - robot
        init_tall init_tiny servingtable cabinet dishwasher countertop - location
        cup mug bowl - item
    )

    (:predicates
        (rob-at ?r - robot ?loc - location)
        (is-at ?obj - item ?loc - location)
        (hand-is-free ?r - robot)
        (restrict-place-to ?loc - location)
        (is-holding ?r - robot ?obj - item)
        (can-reach ?r - robot ?loc - location)
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
        )
        :effect (and
            (is-at ?obj ?loc)
            (not (is-holding ?r ?obj))
            (hand-is-free ?r)
            (increase (total-cost) 100)
        )
    )
    )
    """
    return DOMAIN_PDDL
