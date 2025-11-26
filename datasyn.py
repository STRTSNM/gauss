import heapq
import json
import multiprocessing
import os
import random

import sympy
from sympy import Add, Float, Integer, Mul, Pow, S
from tqdm import tqdm

def get_brain_pain(expr):
    cost = 0
    if isinstance(expr, Integer):
        return 0
    if isinstance(expr, Float):
        return 1

    for arg in expr.args:
        cost += get_brain_pain(arg)

    if isinstance(expr, Add):
        cost += 1
        for arg in expr.args:
            # Reward making 10s (e.g. 10, 20, 30...)
            if isinstance(arg, Integer) and arg % 10 == 0:
                cost -= 0.5

    elif isinstance(expr, Mul):
        # check for Base 10 or Doubling
        is_easy = False
        for arg in expr.args:
            if isinstance(arg, Integer):
                val = abs(int(arg))
                if val in [10, 100, 1000]:
                    is_easy = True
                if val == 2:
                    is_easy = True

        if is_easy:
            cost += 1
        else:
            cost += 10 

    elif isinstance(expr, Pow):
        base, exp = expr.as_base_exp()
        if exp < 0:
            if base == 10:
                cost += 1
            elif base == 2:
                cost += 2
            else:
                cost += 15
        else:
            cost += 10

    return cost

def get_universal_moves(expr):
    moves = set()

  ## Defining the Axioms
    # AXIOM 1: ADDITIVE IDENTITY (n - n)
    # Range limited to 1-3 for speed
    for n in [1, 2, 3]:
        try:
            zero_pair = Add(Integer(n), Mul(Integer(-1), Integer(n)), evaluate=False)
            new_move = Add(expr, zero_pair, evaluate=False)
            moves.add(new_move)
        except:
            pass

    # AXIOM 2: MULTIPLICATIVE IDENTITY (n / n)
    # Range limited to 2, 4, 10
    for n in [2, 4, 10]:
        try:
            one_pair = Mul(Integer(n), Pow(Integer(n), -1), evaluate=False)
            new_move = Mul(expr, one_pair, evaluate=False)
            moves.add(new_move)
        except:
            pass

    # AXIOM 3: ALGEBRA & ARITHMETIC
    try:
        moves.add(sympy.expand(expr))
        moves.add(sympy.factor(expr))
        moves.add(sympy.simplify(expr))
        moves.add(expr.doit())
    except:
        pass

    return list(moves)


def discover_path(start_str):
    steps_limit = 200

    try:
        start_expr = sympy.sympify(start_str, evaluate=False)
    except:
        return None, 0, 0

    start_cost = get_brain_pain(start_expr)

    pq = []
    heapq.heappush(pq, (start_cost, str(start_expr), [str(start_expr)]))

    visited = set()
    best_path = None
    min_pain = start_cost
    steps = 0

    while pq and steps < steps_limit:
        cost_so_far, curr_str, history = heapq.heappop(pq)
        steps += 1

        if curr_str in visited:
            continue
        visited.add(curr_str)

        try:
            curr_expr = sympy.sympify(curr_str, evaluate=False)
        except:
            continue

        current_pain = get_brain_pain(curr_expr)

        if curr_expr.is_Number:
            if current_pain < min_pain:
                min_pain = current_pain
                best_path = history
            continue

        if current_pain > start_cost + 8:
            continue

        moves = get_universal_moves(curr_expr)
        for move in moves:
            if len(str(move)) > 60:
                continue

            move_pain = get_brain_pain(move)
            new_history = history + [str(move)]
            heapq.heappush(pq, (move_pain, str(move), new_history))

    return best_path, min_pain, start_cost


def generate_problem():
    op = random.choice(["add", "sub", "mult", "div", "sq_diff"])

    if op == "add":
        # 18 + 7, 29 + 4
        return f"{random.randint(1, 8) * 10 + random.choice([8, 9])} + {random.randint(2, 9)}"

    elif op == "sub":
        # 53 - 19
        return f"{random.randint(40, 90)} - {random.randint(1, 3) * 10 + 9}"

    elif op == "mult":
        # 16 * 5 or 24 * 5
        return f"{random.randint(2, 20) * 2} * 5"

    elif op == "div":
        # 42 / 5 or 81 / 5
        return f"{random.randint(10, 80)} / 5"

    elif op == "sq_diff":
        # 12^2 - 8^2
        x = random.randint(10, 20)
        return f"{x}**2 - {x - random.randint(1, 4)}**2"

    return "1+1"


def worker_task(x):

    try:
        problem_str = generate_problem()
        path, final_cost, start_cost = discover_path(problem_str)

        if path and final_cost < start_cost:
            clean_chain = " = ".join(path)
            return json.dumps(
                {"prompt": f"{problem_str} =", "completion": f" {clean_chain}"}
            )
    except Exception:
        pass
    return None


if __name__ == "__main__":
    TOTAL_TARGET = 50000
    OUTPUT_FILE = "dataset_fast.jsonl"

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"🚀 Launching Fast Miner on {num_cores} cores...")

    pool = multiprocessing.Pool(processes=num_cores)

    with open(OUTPUT_FILE, "w") as f:
        pbar = tqdm(total=TOTAL_TARGET, unit="ex")
        collected = 0

        for result in pool.imap_unordered(worker_task, range(TOTAL_TARGET * 10)):
            if result:
                f.write(result + "\n")
                collected += 1
                pbar.update(1)

            if collected >= TOTAL_TARGET:
                break

    pool.terminate()
    print(f"\nDone! Saved {TOTAL_TARGET} examples to {OUTPUT_FILE}")
