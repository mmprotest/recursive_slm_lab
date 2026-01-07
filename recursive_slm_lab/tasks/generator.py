from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap


BUNDLED_PATH = Path(__file__).parent / "bundled_tasks.jsonl"


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def _task_id(function_name: str, prompt: str) -> str:
    return f"{function_name}-{_hash_prompt(prompt)}"


def _format_prompt_add(k: int) -> str:
    if k < 0:
        return f"Return n minus {abs(k)}."
    return f"Return n plus {k}."


def _format_prompt_mul(k: int) -> str:
    return f"Return n multiplied by {k}."


def _make_task(
    prompt: str,
    function_name: str,
    signature: str,
    reference_tests: str,
) -> dict[str, str]:
    return {
        "task_id": _task_id(function_name, prompt),
        "prompt": prompt,
        "function_name": function_name,
        "signature": signature,
        "reference_tests": reference_tests,
    }


def _add_const_task(k: int) -> dict[str, str]:
    suffix = f"{k}" if k >= 0 else f"neg{abs(k)}"
    function_name = f"add_const_{suffix}"
    prompt = _format_prompt_add(k)
    reference_tests = textwrap.dedent(
        f"""
        import pytest
        import random
        from solution import *


        def test_examples():
            assert {function_name}(0) == {k}
            assert {function_name}(5) == {5 + k}
            assert {function_name}(-3) == {-3 + k}


        def test_randomized():
            random.seed(1001)
            for _ in range(50):
                n = random.randint(-100, 100)
                assert {function_name}(n) == n + {k}
        """
    ).strip()
    return _make_task(prompt, function_name, "(n)", reference_tests)


def _mul_const_task(k: int) -> dict[str, str]:
    suffix = f"{k}" if k >= 0 else f"neg{abs(k)}"
    function_name = f"mul_const_{suffix}"
    prompt = _format_prompt_mul(k)
    reference_tests = textwrap.dedent(
        f"""
        import pytest
        import random
        from solution import *


        def test_examples():
            assert {function_name}(1) == {k}
            assert {function_name}(3) == {3 * k}
            assert {function_name}(-2) == {-2 * k}


        def test_randomized():
            random.seed(2002)
            for _ in range(50):
                n = random.randint(-50, 50)
                assert {function_name}(n) == n * {k}
        """
    ).strip()
    return _make_task(prompt, function_name, "(n)", reference_tests)


def _rotate_left_task() -> dict[str, str]:
    prompt = "Rotate a list left by one position. Return an empty list for empty input."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def rotate_left_oracle(values):
            if not values:
                return []
            return values[1:] + values[:1]


        def test_examples():
            assert rotate_left([1, 2, 3]) == [2, 3, 1]
            assert rotate_left([]) == []


        def test_randomized():
            random.seed(3003)
            for _ in range(60):
                size = random.randint(0, 12)
                values = [random.randint(-5, 5) for _ in range(size)]
                assert rotate_left(values) == rotate_left_oracle(values)
        """
    ).strip()
    return _make_task(prompt, "rotate_left", "(values)", reference_tests)


def _is_palindrome_alnum_task() -> dict[str, str]:
    prompt = (
        "Return True if text is a palindrome after removing non-alphanumeric characters "
        "and normalizing case."
    )
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        import string
        from solution import *


        def oracle(text):
            filtered = "".join(ch.lower() for ch in text if ch.isalnum())
            return filtered == filtered[::-1]


        def test_examples():
            assert is_palindrome_alnum("A man, a plan, a canal, Panama!") is True
            assert is_palindrome_alnum("Hello") is False


        def test_randomized():
            random.seed(4004)
            alphabet = string.ascii_letters + string.digits + "!?., "
            for _ in range(80):
                size = random.randint(0, 25)
                text = "".join(random.choice(alphabet) for _ in range(size))
                assert is_palindrome_alnum(text) == oracle(text)
        """
    ).strip()
    return _make_task(prompt, "is_palindrome_alnum", "(text)", reference_tests)


def _two_sum_indices_task() -> dict[str, str]:
    prompt = (
        "Given a list of integers and a target, return indices (i, j) with i < j such "
        "that nums[i] + nums[j] == target. Return (-1, -1) if no pair exists."
    )
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle_pair(nums, target):
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    if nums[i] + nums[j] == target:
                        return (i, j)
            return (-1, -1)


        def check_solution(nums, target, result):
            assert isinstance(result, (list, tuple))
            assert len(result) == 2
            i, j = result
            if i == -1 and j == -1:
                assert oracle_pair(nums, target) == (-1, -1)
                return
            assert 0 <= i < j < len(nums)
            assert nums[i] + nums[j] == target


        def test_examples():
            nums = [2, 7, 11, 15]
            result = two_sum_indices(nums, 9)
            check_solution(nums, 9, result)
            assert two_sum_indices([1, 2, 3], 99) == (-1, -1)


        def test_randomized():
            random.seed(5005)
            for _ in range(80):
                size = random.randint(0, 12)
                nums = [random.randint(-10, 10) for _ in range(size)]
                target = random.randint(-10, 10)
                result = two_sum_indices(nums, target)
                check_solution(nums, target, result)
        """
    ).strip()
    return _make_task(prompt, "two_sum_indices", "(nums, target)", reference_tests)


def _clamp_task() -> dict[str, str]:
    prompt = "Clamp x to the inclusive range [lo, hi]."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def test_examples():
            assert clamp(5, 0, 3) == 3
            assert clamp(-1, 0, 3) == 0
            assert clamp(2, 0, 3) == 2


        def test_randomized():
            random.seed(6006)
            for _ in range(80):
                lo = random.randint(-10, 5)
                hi = random.randint(lo, 10)
                x = random.randint(-20, 20)
                assert clamp(x, lo, hi) == min(max(x, lo), hi)
        """
    ).strip()
    return _make_task(prompt, "clamp", "(x, lo, hi)", reference_tests)


def _dedupe_preserve_task() -> dict[str, str]:
    prompt = "Remove duplicate items from a list while preserving the first occurrence order."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values):
            seen = set()
            result = []
            for item in values:
                if item in seen:
                    continue
                seen.add(item)
                result.append(item)
            return result


        def test_examples():
            assert dedupe_preserve([1, 2, 1, 3, 2]) == [1, 2, 3]
            assert dedupe_preserve([]) == []


        def test_randomized():
            random.seed(7007)
            for _ in range(80):
                size = random.randint(0, 15)
                values = [random.randint(0, 6) for _ in range(size)]
                assert dedupe_preserve(values) == oracle(values)
        """
    ).strip()
    return _make_task(prompt, "dedupe_preserve", "(values)", reference_tests)


def _parse_int_list_task() -> dict[str, str]:
    prompt = (
        "Parse a string of integers separated by commas and/or whitespace. "
        "Return an empty list for empty or whitespace-only input."
    )
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(text):
            if not text.strip():
                return []
            normalized = text.replace(",", " ")
            parts = [p for p in normalized.split() if p]
            return [int(p) for p in parts]


        def test_examples():
            assert parse_int_list("1, 2,3") == [1, 2, 3]
            assert parse_int_list("  -1  4   5 ") == [-1, 4, 5]
            assert parse_int_list("") == []


        def test_randomized():
            random.seed(8008)
            separators = [",", ", ", " ", "  ", " , "]
            for _ in range(80):
                size = random.randint(0, 10)
                values = [random.randint(-20, 20) for _ in range(size)]
                if not values:
                    text = ""
                else:
                    text = str(values[0])
                    for value in values[1:]:
                        text += random.choice(separators) + str(value)
                assert parse_int_list(text) == oracle(text)
        """
    ).strip()
    return _make_task(prompt, "parse_int_list", "(text)", reference_tests)


def _max_subarray_sum_task() -> dict[str, str]:
    prompt = "Return the maximum sum over all contiguous subarrays. Return 0 for an empty list."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values):
            if not values:
                return 0
            best = None
            for i in range(len(values)):
                total = 0
                for j in range(i, len(values)):
                    total += values[j]
                    if best is None or total > best:
                        best = total
            return best if best is not None else 0


        def test_examples():
            assert max_subarray_sum([1, -2, 3, 4]) == 7
            assert max_subarray_sum([-5, -1, -3]) == -1
            assert max_subarray_sum([]) == 0


        def test_randomized():
            random.seed(9009)
            for _ in range(60):
                size = random.randint(0, 30)
                values = [random.randint(-10, 10) for _ in range(size)]
                assert max_subarray_sum(values) == oracle(values)
        """
    ).strip()
    return _make_task(prompt, "max_subarray_sum", "(values)", reference_tests)


def _bfs_distance_task() -> dict[str, str]:
    prompt = (
        "Given an unweighted graph as an adjacency list, return the shortest path length "
        "from start to goal using BFS. Return -1 if goal is unreachable."
    )
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from collections import deque
        from solution import *


        def oracle(graph, start, goal):
            if start == goal:
                return 0
            visited = {start}
            queue = deque([(start, 0)])
            while queue:
                node, dist = queue.popleft()
                for neighbor in graph.get(node, []):
                    if neighbor in visited:
                        continue
                    if neighbor == goal:
                        return dist + 1
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
            return -1


        def test_examples():
            graph = {0: [1, 2], 1: [2], 2: [3], 3: []}
            assert bfs_distance(graph, 0, 3) == 2
            assert bfs_distance(graph, 3, 0) == -1
            assert bfs_distance(graph, 1, 1) == 0


        def test_randomized():
            random.seed(1010)
            for _ in range(60):
                n = random.randint(2, 15)
                graph = {i: [] for i in range(n)}
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.2:
                            graph[i].append(j)
                            graph[j].append(i)
                start = random.randrange(n)
                goal = random.randrange(n)
                assert bfs_distance(graph, start, goal) == oracle(graph, start, goal)
        """
    ).strip()
    return _make_task(prompt, "bfs_distance", "(graph, start, goal)", reference_tests)


def _interleave_range(start: int, stop: int) -> list[int]:
    values = []
    for k in range(start, stop + 1):
        values.append(k)
        values.append(-k)
    return values


def generate_tasks(count: int) -> list[dict[str, str]]:
    if count < 120:
        raise ValueError("count must be at least 120")

    core_tasks = [
        _rotate_left_task(),
        _is_palindrome_alnum_task(),
        _two_sum_indices_task(),
        _clamp_task(),
        _dedupe_preserve_task(),
        _parse_int_list_task(),
        _max_subarray_sum_task(),
        _bfs_distance_task(),
    ]

    add_values = _interleave_range(1, 120)
    mul_values = _interleave_range(2, 40)

    required_add = add_values[:80]
    required_mul = mul_values[:30]
    extra_add = add_values[80:]
    extra_mul = mul_values[30:]

    tasks = core_tasks
    tasks += [_add_const_task(k) for k in required_add]
    tasks += [_mul_const_task(k) for k in required_mul]

    if len(tasks) < count:
        extras = [_add_const_task(k) for k in extra_add] + [_mul_const_task(k) for k in extra_mul]
        needed = count - len(tasks)
        tasks += extras[:needed]

    return tasks[:count]


def write_tasks(tasks: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate bundled tasks JSONL.")
    parser.add_argument("--count", type=int, default=200, help="Number of tasks to generate")
    parser.add_argument("--out", type=Path, default=BUNDLED_PATH, help="Output JSONL path")
    args = parser.parse_args(argv)

    tasks = generate_tasks(args.count)
    write_tasks(tasks, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
