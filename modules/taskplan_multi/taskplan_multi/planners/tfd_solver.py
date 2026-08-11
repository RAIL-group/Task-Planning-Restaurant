"""
Thin subprocess wrapper around Temporal Fast Downward (neighthan/tfd, a
python3 port of TFD v0.4 / IPC 2014), invoked via the `downward/tfd`
entry point installed at $TFD_ROOT in the Docker image (see Dockerfile).

Unlike taskplan_multi.planners.myopic_planner / decentralized_planner,
which call pddlstream.algorithms.search.solve_from_pddl (a classical,
non-temporal Fast Downward build that cannot parse :durative-action
domains at all), this goes around pddlstream entirely: TFD is a separate
solver with its own PDDL2.1 durative-action support, its own CLI, and its
own plan-file format -- there's no shared interface to reuse here.
"""

import glob
import os
import re
import subprocess
import tempfile

TFD_ROOT = os.environ.get('TFD_ROOT', '/tfd/downward')

# TFD's own default IPC-2008 config is "y+Y+a+e+r+O+1+C+1+b". The 'a' flag
# is anytime_search: with it set, TFD doesn't stop at the first valid plan,
# it keeps searching for better ones until the time budget is exhausted --
# on every single planning call. We only need *a* plan (satisficing, not
# optimal), so 'a' is dropped; everything else is left alone since 'y'/'Y'
# select the heuristic TFD needs to search at all.
TFD_CONFIG = 'y+Y+e+r+O+1+C+1+b'

# Standard IPC PDDL2.1 temporal-plan line: "<start>: (name args...) [<duration>]"
_PLAN_LINE = re.compile(r'^\s*([\d.]+)\s*:\s*\(\s*([^)]*?)\s*\)\s*\[\s*([\d.]+)\s*\]\s*$')


class TFDSolveError(RuntimeError):
    pass


def solve_temporal_pddl(domain_pddl, problem_pddl, max_planner_time=120):
    """
    Run TFD on a durative-action domain/problem pair. Returns a list of
    (start_time, name, args, duration) tuples in start-time order -- the
    actual concurrent schedule TFD found -- or None if no plan exists.

    TFD writes successively improving solutions as <result>.1, <result>.2,
    ...; with anytime_search off (see TFD_CONFIG) there's normally just
    one. If max_planner_time is hit anyway (a much slower-than-expected
    solve), whatever's been written so far is still used rather than
    thrown away -- an anytime planner killed mid-search often still left a
    valid, just not-yet-improved-on, solution on disk.
    """
    tfd_script = os.path.join(TFD_ROOT, 'tfd')
    if not os.path.exists(tfd_script):
        raise TFDSolveError(
            f"TFD entry point not found at {tfd_script}. This solver only "
            f"runs inside the project's Docker image (see Dockerfile); "
            f"set TFD_ROOT if it's installed somewhere else.")

    with tempfile.TemporaryDirectory() as tmp:
        domain_path = os.path.join(tmp, 'domain.pddl')
        problem_path = os.path.join(tmp, 'problem.pddl')
        result_path = os.path.join(tmp, 'plan')

        with open(domain_path, 'w') as f:
            f.write(domain_pddl)
        with open(problem_path, 'w') as f:
            f.write(problem_pddl)

        try:
            proc = subprocess.run(
                [tfd_script, domain_path, problem_path, result_path, TFD_CONFIG],
                cwd=tmp,
                timeout=max_planner_time,
                capture_output=True,
                text=True,
            )
            returncode = proc.returncode
            stdout, stderr = proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            returncode = None
            stdout = (exc.stdout or b'').decode() if isinstance(exc.stdout, bytes) else (exc.stdout or '')
            stderr = (exc.stderr or b'').decode() if isinstance(exc.stderr, bytes) else (exc.stderr or '')

        solution_path = _best_solution_file(result_path)
        if solution_path is None:
            if returncode is None:
                raise TFDSolveError(
                    f"TFD didn't finish within {max_planner_time}s and left "
                    f"no solution file.\nstdout:\n{stdout}\nstderr:\n{stderr}")
            if returncode != 0:
                raise TFDSolveError(
                    f"TFD exited with code {returncode} and produced "
                    f"no solution file.\nstdout:\n{stdout}\n"
                    f"stderr:\n{stderr}")
            return None

        return _parse_plan(solution_path)


def _best_solution_file(result_path):
    numbered = glob.glob(f'{result_path}.*')
    if numbered:
        return max(numbered, key=_solution_number)
    if os.path.exists(result_path):
        return result_path
    return None


def _solution_number(path):
    try:
        return int(path.rsplit('.', 1)[-1])
    except ValueError:
        return -1


def _parse_plan(path):
    steps = []
    with open(path) as f:
        raw = f.read()
    for line in raw.splitlines():
        match = _PLAN_LINE.match(line)
        if not match:
            continue
        start_time = float(match.group(1))
        tokens = match.group(2).split()
        if not tokens:
            continue
        name, args = tokens[0], tuple(tokens[1:])
        duration = float(match.group(3))
        steps.append((start_time, name, args, duration))

    if not steps:
        raise TFDSolveError(
            f"TFD wrote a solution file but no line matched the expected "
            f"'<time>: (action args) [<duration>]' format -- raw "
            f"content:\n{raw}")

    steps.sort(key=lambda s: s[0])
    return steps
