import os
import taskplan_multi
import pddlstream
import pddlstream.algorithms.meta
import pddlstream.language.constants
from pddlstream.algorithms.search import solve_from_pddl
import time

FAILED_COST = 2000


class MyopicPlanner:
    def __init__(self, domain=taskplan_multi.pddl.domain.get_domain(), args=None):
        self.domain = domain

    def get_cost_and_state_from_task(self, proc_data, task):
        pddl_problem = taskplan_multi.pddl.problem.get_problem(proc_data, task)
        planner = 'ff-wastar2'
        plan, cost = solve_from_pddl(
            self.domain,
            pddl_problem,
            planner=planner,
            max_planner_time=300
        )
        
        if plan:
            move_cost = 0
            move_plans = [p for p in plan if p.name == "move"]
            for move in move_plans:
                src = move.args[1]
                target = move.args[2]
                move_cost += proc_data.known_cost[src][target]
            cost += move_cost

        return plan, cost

        # print(move_cost)
        # if plan is None:
        #     proc_data.active_all = True
        #     pddl_problem = taskplan_multi.pddl.problem.get_problem(proc_data, task)
        #     mod_plan, cost = solve_from_pddl(
        #         self.domain,
        #         pddl_problem,
        #         planner=planner,
        #         max_planner_time=120
        #     )
        #     proc_data.active_all = False
        #     return mod_plan, cost + 2000
        # else:
        # print(cost)
        # print(move_cost)
        # cost += move_cost
        # return plan, cost
    
    def get_timed_plan(self, proc_data, task):
        """
        Wraps get_cost_and_state_from_task and annotates each action with
        its start and end time, using PDDL action costs as durations:
            move       -> known_cost[src][dst]
            pick/place/mix/serve -> 10
            wash/restock         -> 30
            ask-help             -> 500

        Returns (steps, pddl_plan, total_cost) where steps is a list of dicts:
            {
                't_start':     int,
                't_end':       int,
                'action':      str,   # action name
                'args':        tuple, # raw PDDL args
                'description': str,   # human-readable summary
            }
        pddl_plan is the raw plan list from the solver (used for broadcast-state
        projection in concurrent simulations).
        Returns (None, None, cost) if no plan is found.
        """
        _FIXED_COST = {
            'pick': 10, 'place': 10, 'mix': 10, 'serve': 10,
            'wash': 30, 'restock': 30, 'ask-help': 500,
        }

        plan, cost = self.get_cost_and_state_from_task(proc_data, task)
        if plan is None:
            return None, None, cost

        steps = []
        t = 0
        for action in plan:
            name, args = action.name, action.args

            if name == 'move':
                robot, src, dst = args[0], args[1], args[2]
                duration = proc_data.known_cost[src][dst]
                desc = f"{robot} moves {src} -> {dst}"
            elif name == 'pick':
                robot, obj, loc = args[0], args[1], args[2]
                duration = _FIXED_COST['pick']
                desc = f"{robot} picks {obj} from {loc}"
            elif name == 'place':
                robot, obj, loc = args[0], args[1], args[2]
                duration = _FIXED_COST['place']
                desc = f"{robot} places {obj} at {loc}"
            elif name == 'wash':
                robot, item = args[0], args[1]
                duration = _FIXED_COST['wash']
                desc = f"{robot} washes {item}"
            elif name == 'mix':
                robot, item, bowl, loc = args[0], args[1], args[2], args[3]
                duration = _FIXED_COST['mix']
                desc = f"{robot} mixes {item} into {bowl} at {loc}"
            elif name == 'serve':
                robot, item, bowl, loc = args[0], args[1], args[2], args[3]
                duration = _FIXED_COST['serve']
                desc = f"{robot} serves {item} in {bowl} at {loc}"
            elif name == 'restock':
                robot, item = args[0], args[1]
                duration = _FIXED_COST['restock']
                desc = f"{robot} restocks {item}"
            elif name == 'ask-help':
                robot = args[0]
                duration = _FIXED_COST['ask-help']
                desc = f"ask {robot} for help"
            else:
                duration = 0
                desc = f"{name} {' '.join(str(a) for a in args)}"

            steps.append({
                't_start': t,
                't_end': t + duration,
                'action': name,
                'args': args,
                'description': desc,
            })
            t += duration

        return steps, plan, cost

    def get_expected_cost(self, proc_data, task_distribution):
        expected_costs = list()
        for task in task_distribution:
            plan, cost = self.get_cost_and_state_from_task(proc_data, task)
            if plan is None:
                expected_costs.append(FAILED_COST)
            else:
                expected_costs.append(cost)
        # print(expected_costs)
        expected_cost = sum(expected_costs)/len(expected_costs)
        return expected_cost
    
    def get_oracle_failure_ratio(self, restaurant, task_seq):
        failed = 0
        for idx, item in enumerate(task_seq):
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                if plan is None:
                    failed+=1
        return failed
    
    def get_seq_cost(self, args, restaurant, task_seq, seq_num, no_prep_state=None, prep_state=None, ap=None):
        if no_prep_state:
            restaurant.update_container_props(no_prep_state)
            file_name = 'np_myopic.txt'
            logfile = os.path.join(args.save_dir, file_name)
            # task_file_name = 'np_myopic_tasks.txt'
            # logfile_task = os.path.join(args.save_dir, task_file_name)
            for idx, item in enumerate(task_seq):
                start = time.time()
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                restaurant.asked_help = False
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                end = time.time()
                elapsed = end - start
                if plan is None:
                    # costs.append(10000)
                    with open(logfile, "a+") as f:
                        f.write(
                            f" | seq: S{seq_num}"
                            f" | num: T{idx+1}"
                            f" | active: {active_agent}"
                            f" | time: {elapsed:0.2f}"
                            f" | help: 3"
                            f" | cost: 2000 \n"
                        )
                    continue
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | active: {active_agent}"
                        f" | time: {elapsed:0.2f}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.2f} \n"
                    )
                # with open(logfile_task, "a+") as f:
                #     f.write(
                #         f" | num: T{idx+1}"
                #         f" | task: {task}"
                #         f" | plan: {plan}\n"
                #     )
                # costs.append(cost)
                new_state = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(new_state)
                # taskplan_multi.utils.plot_state(restaurant, args, image_name=f'np-mp-{seq_num}-T{idx+1}', title=f'State after Task {idx+1}')
        
        if prep_state:
            restaurant.update_container_props(prep_state)
            file_name = 'prep_myopic.txt'
            logfile = os.path.join(args.save_dir, file_name)
            for idx, item in enumerate(task_seq):
                start = time.time()
                if ap is not None:
                    if (idx + 1) % 2 == 1:
                        new_prep_state = ap.get_prepared_state_by_cleaner(restaurant, n_iterations=50)
                        if new_prep_state is not None:
                            restaurant.update_container_props(new_prep_state)
                    # ap.concern = 'joint'
                active_agent = item[0]
                task = item[1]
                restaurant.active_robot = active_agent
                plan, cost = (
                    self.get_cost_and_state_from_task(
                        restaurant, task)
                )
                end = time.time()
                elapsed = end - start
                if plan is None:
                    # costs.append(10000)
                    with open(logfile, "a+") as f:
                        f.write(
                            f" | seq: S{seq_num}"
                            f" | num: T{idx+1}"
                            f" | active: {active_agent}"
                            f" | time: {elapsed:0.2f}"
                            f" | help: 3"
                            f" | cost: 2000 \n"
                        )
                    continue
                help_stat = taskplan_multi.utils.get_status_of_asking_help(plan)
                with open(logfile, "a+") as f:
                    f.write(
                        f" | seq: S{seq_num}"
                        f" | num: T{idx+1}"
                        f" | active: {active_agent}"
                        f" | time: {elapsed:0.2f}"
                        f" | help: {help_stat}"
                        f" | cost: {cost:0.2f} \n"
                    )
                new_state = restaurant.get_final_state_from_plan(plan)
                restaurant.update_container_props(new_state)
                # taskplan_multi.utils.plot_state(restaurant, args, image_name=f'prep-mp-{seq_num}-T{idx+1}', title=f'State after Task {idx+1}')
