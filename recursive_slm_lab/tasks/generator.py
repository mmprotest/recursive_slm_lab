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
    assert_tests: list[str],
    category: str = "misc",
    difficulty: int = 1,
    tags: list[str] | None = None,
) -> dict[str, str | list[str]]:
    payload: dict[str, str | list[str]] = {
        "task_id": _task_id(function_name, prompt),
        "prompt": prompt,
        "function_name": function_name,
        "signature": signature,
        "reference_tests": reference_tests,
        "category": category,
        "difficulty": difficulty,
        "assert_tests": assert_tests,
    }
    if tags:
        payload["tags"] = tags
    return payload


def _add_const_task(k: int) -> dict[str, str | list[str]]:
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
    assert_tests = [
        f"assert {function_name}(0) == {k}",
        f"assert {function_name}(5) == {5 + k}",
        f"assert {function_name}(-3) == {-3 + k}",
    ]
    return _make_task(prompt, function_name, "(n)", reference_tests, assert_tests, category="math", difficulty=1)


def _mul_const_task(k: int) -> dict[str, str | list[str]]:
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
    assert_tests = [
        f"assert {function_name}(1) == {k}",
        f"assert {function_name}(3) == {3 * k}",
        f"assert {function_name}(-2) == {-2 * k}",
    ]
    return _make_task(prompt, function_name, "(n)", reference_tests, assert_tests, category="math", difficulty=1)


def _rotate_left_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert rotate_left([1, 2, 3]) == [2, 3, 1]",
        "assert rotate_left([]) == []",
    ]
    return _make_task(prompt, "rotate_left", "(values)", reference_tests, assert_tests, category="lists", difficulty=1)


def _is_palindrome_alnum_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert is_palindrome_alnum('A man, a plan, a canal, Panama!') is True",
        "assert is_palindrome_alnum('Hello') is False",
    ]
    return _make_task(
        prompt, "is_palindrome_alnum", "(text)", reference_tests, assert_tests, category="strings", difficulty=2
    )


def _two_sum_indices_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert two_sum_indices([2, 7, 11, 15], 9) == (0, 1)",
        "assert two_sum_indices([1, 2, 3], 99) == (-1, -1)",
    ]
    return _make_task(
        prompt, "two_sum_indices", "(nums, target)", reference_tests, assert_tests, category="algorithms", difficulty=2
    )


def _clamp_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert clamp(5, 0, 3) == 3",
        "assert clamp(-1, 0, 3) == 0",
        "assert clamp(2, 0, 3) == 2",
    ]
    return _make_task(prompt, "clamp", "(x, lo, hi)", reference_tests, assert_tests, category="math", difficulty=1)


def _gcd_task() -> dict[str, str | list[str]]:
    prompt = "Return the greatest common divisor (GCD) of two integers."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(a, b):
            a, b = abs(a), abs(b)
            while b:
                a, b = b, a % b
            return a


        def test_examples():
            assert gcd_value(12, 18) == 6
            assert gcd_value(-10, 5) == 5


        def test_randomized():
            random.seed(1901)
            for _ in range(60):
                a = random.randint(-50, 50)
                b = random.randint(-50, 50)
                assert gcd_value(a, b) == oracle(a, b)
        """
    ).strip()
    assert_tests = [
        "assert gcd_value(12, 18) == 6",
        "assert gcd_value(-10, 5) == 5",
    ]
    return _make_task(
        prompt,
        "gcd_value",
        "(a, b)",
        reference_tests,
        assert_tests,
        category="math",
        difficulty=2,
    )


def _lcm_task() -> dict[str, str | list[str]]:
    prompt = "Return the least common multiple (LCM) of two integers."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def gcd(a, b):
            a, b = abs(a), abs(b)
            while b:
                a, b = b, a % b
            return a


        def oracle(a, b):
            if a == 0 or b == 0:
                return 0
            return abs(a * b) // gcd(a, b)


        def test_examples():
            assert lcm_value(4, 6) == 12
            assert lcm_value(0, 5) == 0


        def test_randomized():
            random.seed(1902)
            for _ in range(60):
                a = random.randint(-20, 20)
                b = random.randint(-20, 20)
                assert lcm_value(a, b) == oracle(a, b)
        """
    ).strip()
    assert_tests = [
        "assert lcm_value(4, 6) == 12",
        "assert lcm_value(0, 5) == 0",
    ]
    return _make_task(
        prompt,
        "lcm_value",
        "(a, b)",
        reference_tests,
        assert_tests,
        category="math",
        difficulty=2,
    )


def _piecewise_task() -> dict[str, str | list[str]]:
    prompt = "Return x*x if x < 0, return x if 0 <= x <= 10, else return 10."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(x):
            if x < 0:
                return x * x
            if x <= 10:
                return x
            return 10


        def test_examples():
            assert piecewise_value(-3) == 9
            assert piecewise_value(5) == 5
            assert piecewise_value(12) == 10


        def test_randomized():
            random.seed(1903)
            for _ in range(50):
                x = random.randint(-10, 20)
                assert piecewise_value(x) == oracle(x)
        """
    ).strip()
    assert_tests = [
        "assert piecewise_value(-3) == 9",
        "assert piecewise_value(12) == 10",
    ]
    return _make_task(
        prompt,
        "piecewise_value",
        "(x)",
        reference_tests,
        assert_tests,
        category="math",
        difficulty=1,
    )


def _dedupe_preserve_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert dedupe_preserve([1, 2, 1, 3, 2]) == [1, 2, 3]",
        "assert dedupe_preserve([]) == []",
    ]
    return _make_task(
        prompt, "dedupe_preserve", "(values)", reference_tests, assert_tests, category="lists", difficulty=1
    )


def _parse_int_list_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert parse_int_list('1, 2,3') == [1, 2, 3]",
        "assert parse_int_list('  -1  4   5 ') == [-1, 4, 5]",
        "assert parse_int_list('') == []",
    ]
    return _make_task(prompt, "parse_int_list", "(text)", reference_tests, assert_tests, category="parsing", difficulty=1)


def _flatten_one_level_task() -> dict[str, str | list[str]]:
    prompt = "Flatten a list by one level (if an element is a list, extend it; otherwise keep it)."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values):
            flattened = []
            for item in values:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            return flattened


        def test_examples():
            assert flatten_one_level([1, [2, 3], 4]) == [1, 2, 3, 4]
            assert flatten_one_level([]) == []


        def test_randomized():
            random.seed(1801)
            for _ in range(50):
                values = []
                for _ in range(random.randint(0, 8)):
                    if random.random() < 0.4:
                        values.append([random.randint(0, 5) for _ in range(random.randint(0, 4))])
                    else:
                        values.append(random.randint(0, 5))
                assert flatten_one_level(values) == oracle(values)
        """
    ).strip()
    assert_tests = [
        "assert flatten_one_level([1, [2, 3], 4]) == [1, 2, 3, 4]",
        "assert flatten_one_level([]) == []",
    ]
    return _make_task(
        prompt,
        "flatten_one_level",
        "(values)",
        reference_tests,
        assert_tests,
        category="lists",
        difficulty=2,
    )


def _max_subarray_sum_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert max_subarray_sum([1, -2, 3, 4]) == 7",
        "assert max_subarray_sum([-5, -1, -3]) == -1",
        "assert max_subarray_sum([]) == 0",
    ]
    return _make_task(
        prompt, "max_subarray_sum", "(values)", reference_tests, assert_tests, category="algorithms", difficulty=2
    )


def _bfs_distance_task() -> dict[str, str | list[str]]:
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
    assert_tests = [
        "assert bfs_distance({0: [1, 2], 1: [2], 2: [3], 3: []}, 0, 3) == 2",
        "assert bfs_distance({0: [1, 2], 1: [2], 2: [3], 3: []}, 3, 0) == -1",
        "assert bfs_distance({0: [1], 1: []}, 0, 0) == 0",
    ]
    return _make_task(
        prompt,
        "bfs_distance",
        "(graph, start, goal)",
        reference_tests,
        assert_tests,
        category="algorithms",
        difficulty=3,
    )


def _parse_date_iso_task() -> dict[str, str | list[str]]:
    prompt = (
        "Normalize date strings in either YYYY-MM-DD or YYYY/MM/DD format to YYYY-MM-DD."
    )
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(text):
            text = text.strip()
            if "-" in text:
                parts = text.split("-")
            else:
                parts = text.split("/")
            year, month, day = [p.zfill(2) for p in parts]
            year = year.zfill(4)
            return f"{year}-{month}-{day}"


        def test_examples():
            assert normalize_date("2024/3/5") == "2024-03-05"
            assert normalize_date("1999-12-1") == "1999-12-01"


        def test_randomized():
            random.seed(1111)
            for _ in range(50):
                year = random.randint(1990, 2030)
                month = random.randint(1, 12)
                day = random.randint(1, 28)
                if random.random() < 0.5:
                    text = f"{year}/{month}/{day}"
                else:
                    text = f"{year}-{month}-{day}"
                assert normalize_date(text) == oracle(text)
        """
    ).strip()
    assert_tests = [
        "assert normalize_date('2024/3/5') == '2024-03-05'",
        "assert normalize_date('1999-12-1') == '1999-12-01'",
    ]
    return _make_task(
        prompt,
        "normalize_date",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=1,
        tags=["heldout_only"],
    )


def _invert_mapping_task() -> dict[str, str | list[str]]:
    prompt = "Invert a dict so values map to sorted lists of keys. Ignore None values."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def oracle(mapping):
            inverted = {}
            for key, value in mapping.items():
                if value is None:
                    continue
                inverted.setdefault(value, []).append(key)
            for value in inverted:
                inverted[value].sort()
            return inverted


        def test_examples():
            assert invert_mapping({"a": 1, "b": 1, "c": 2}) == {1: ["a", "b"], 2: ["c"]}
            assert invert_mapping({}) == {}
            assert invert_mapping({"x": None}) == {}
        """
    ).strip()
    assert_tests = [
        "assert invert_mapping({'a': 1, 'b': 1, 'c': 2}) == {1: ['a', 'b'], 2: ['c']}",
        "assert invert_mapping({}) == {}",
    ]
    return _make_task(
        prompt,
        "invert_mapping",
        "(mapping)",
        reference_tests,
        assert_tests,
        category="dicts",
        difficulty=2,
    )


def _merge_with_precedence_task() -> dict[str, str | list[str]]:
    prompt = "Merge two dicts; values from the second dict override the first."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert merge_with_precedence({"a": 1, "b": 2}, {"b": 3, "c": 4}) == {"a": 1, "b": 3, "c": 4}
            assert merge_with_precedence({}, {"x": 1}) == {"x": 1}
        """
    ).strip()
    assert_tests = [
        "assert merge_with_precedence({'a': 1, 'b': 2}, {'b': 3, 'c': 4}) == {'a': 1, 'b': 3, 'c': 4}",
        "assert merge_with_precedence({}, {'x': 1}) == {'x': 1}",
    ]
    return _make_task(
        prompt,
        "merge_with_precedence",
        "(first, second)",
        reference_tests,
        assert_tests,
        category="dicts",
        difficulty=1,
    )


def _parse_key_value_task() -> dict[str, str | list[str]]:
    prompt = "Parse a comma-separated list of key=value pairs into a dict of ints. Ignore empty items."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_kv_pairs("a=1,b=2") == {"a": 1, "b": 2}
            assert parse_kv_pairs(" x=3 , y=4 ") == {"x": 3, "y": 4}
            assert parse_kv_pairs("") == {}
        """
    ).strip()
    assert_tests = [
        "assert parse_kv_pairs('a=1,b=2') == {'a': 1, 'b': 2}",
        "assert parse_kv_pairs('') == {}",
    ]
    return _make_task(
        prompt,
        "parse_kv_pairs",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=1,
        tags=["heldout_only"],
    )


def _parse_csv_row_task() -> dict[str, str | list[str]]:
    prompt = "Parse a CSV row with commas (no quoted fields). Return a list of trimmed fields."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_csv_row("a,b,c") == ["a", "b", "c"]
            assert parse_csv_row(" 1, 2 ,3 ") == ["1", "2", "3"]
            assert parse_csv_row("") == [""]
        """
    ).strip()
    assert_tests = [
        "assert parse_csv_row('a,b,c') == ['a', 'b', 'c']",
        "assert parse_csv_row(' 1, 2 ,3 ') == ['1', '2', '3']",
    ]
    return _make_task(prompt, "parse_csv_row", "(text)", reference_tests, assert_tests, category="parsing", difficulty=1)


def _parse_csv_quoted_task() -> dict[str, str | list[str]]:
    prompt = "Parse a CSV row that may include quoted commas. Return a list of fields."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import csv
        from solution import *


        def oracle(text):
            return next(csv.reader([text]))


        def test_examples():
            assert parse_csv_quoted('a,"b,c",d') == ["a", "b,c", "d"]
            assert parse_csv_quoted('"x""y",z') == ['x"y', "z"]
        """
    ).strip()
    assert_tests = [
        "assert parse_csv_quoted('a,\"b,c\",d') == ['a', 'b,c', 'd']",
        "assert parse_csv_quoted('\"x\"\"y\",z') == ['x\"y', 'z']",
    ]
    return _make_task(
        prompt,
        "parse_csv_quoted",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=3,
        tags=["heldout_only"],
    )


def _parse_env_block_task() -> dict[str, str | list[str]]:
    prompt = "Parse lines of KEY=VALUE into a dict. Ignore blank lines and lines without '='."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            text = "HOST=localhost\\nPORT=8080\\n\\nINVALID"
            assert parse_env_block(text) == {"HOST": "localhost", "PORT": "8080"}
            assert parse_env_block("") == {}
        """
    ).strip()
    assert_tests = [
        "assert parse_env_block('HOST=localhost\\nPORT=8080') == {'HOST': 'localhost', 'PORT': '8080'}",
        "assert parse_env_block('') == {}",
    ]
    return _make_task(
        prompt,
        "parse_env_block",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=2,
        tags=["heldout_only"],
    )


def _parse_range_list_task() -> dict[str, str | list[str]]:
    prompt = "Parse a list like '1-3,5,7-8' into sorted integers [1,2,3,5,7,8]."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_ranges("1-3,5,7-8") == [1, 2, 3, 5, 7, 8]
            assert parse_ranges("") == []
            assert parse_ranges("4") == [4]
        """
    ).strip()
    assert_tests = [
        "assert parse_ranges('1-3,5,7-8') == [1, 2, 3, 5, 7, 8]",
        "assert parse_ranges('') == []",
    ]
    return _make_task(
        prompt,
        "parse_ranges",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=2,
        tags=["heldout_only"],
    )


def _parse_bool_tokens_task() -> dict[str, str | list[str]]:
    prompt = "Parse tokens like 'true,false,1,0,yes,no' into booleans. Ignore empty tokens."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_bool_tokens("true,false,1,0") == [True, False, True, False]
            assert parse_bool_tokens("yes, no, TRUE") == [True, False, True]
        """
    ).strip()
    assert_tests = [
        "assert parse_bool_tokens('true,false,1,0') == [True, False, True, False]",
        "assert parse_bool_tokens('yes, no, TRUE') == [True, False, True]",
    ]
    return _make_task(
        prompt, "parse_bool_tokens", "(text)", reference_tests, assert_tests, category="parsing", difficulty=1
    )


def _parse_int_matrix_task() -> dict[str, str | list[str]]:
    prompt = "Parse rows separated by ';' with comma-separated integers into a matrix."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_int_matrix("1,2;3,4") == [[1, 2], [3, 4]]
            assert parse_int_matrix("5") == [[5]]
            assert parse_int_matrix("") == []
        """
    ).strip()
    assert_tests = [
        "assert parse_int_matrix('1,2;3,4') == [[1, 2], [3, 4]]",
        "assert parse_int_matrix('') == []",
    ]
    return _make_task(
        prompt, "parse_int_matrix", "(text)", reference_tests, assert_tests, category="parsing", difficulty=2
    )


def _normalize_whitespace_task() -> dict[str, str | list[str]]:
    prompt = "Normalize whitespace by collapsing runs to single spaces and trimming."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert normalize_whitespace("  hello   world ") == "hello world"
            assert normalize_whitespace("\\n\\tfoo\\tbar  ") == "foo bar"
        """
    ).strip()
    assert_tests = [
        "assert normalize_whitespace('  hello   world ') == 'hello world'",
        "assert normalize_whitespace('\\n\\tfoo\\tbar  ') == 'foo bar'",
    ]
    return _make_task(
        prompt, "normalize_whitespace", "(text)", reference_tests, assert_tests, category="strings", difficulty=1
    )


def _parse_duration_minutes_task() -> dict[str, str | list[str]]:
    prompt = "Convert a duration like '2h 30m' or '45m' into total minutes."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert duration_to_minutes("2h 30m") == 150
            assert duration_to_minutes("45m") == 45
            assert duration_to_minutes("1h") == 60
        """
    ).strip()
    assert_tests = [
        "assert duration_to_minutes('2h 30m') == 150",
        "assert duration_to_minutes('45m') == 45",
    ]
    return _make_task(
        prompt,
        "duration_to_minutes",
        "(text)",
        reference_tests,
        assert_tests,
        category="parsing",
        difficulty=2,
        tags=["heldout_only"],
    )


def _parse_scored_pairs_task() -> dict[str, str | list[str]]:
    prompt = "Parse 'name:score' pairs separated by commas into a dict of ints."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert parse_scored_pairs("ann:1,bob:3") == {"ann": 1, "bob": 3}
            assert parse_scored_pairs("") == {}
        """
    ).strip()
    assert_tests = [
        "assert parse_scored_pairs('ann:1,bob:3') == {'ann': 1, 'bob': 3}",
        "assert parse_scored_pairs('') == {}",
    ]
    return _make_task(
        prompt, "parse_scored_pairs", "(text)", reference_tests, assert_tests, category="parsing", difficulty=1
    )


def _merge_intervals_task() -> dict[str, str | list[str]]:
    prompt = "Merge overlapping intervals and return a list of non-overlapping intervals sorted by start."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(intervals):
            if not intervals:
                return []
            intervals = sorted(intervals, key=lambda x: x[0])
            merged = [intervals[0][:]]
            for start, end in intervals[1:]:
                last = merged[-1]
                if start <= last[1]:
                    last[1] = max(last[1], end)
                else:
                    merged.append([start, end])
            return merged


        def test_examples():
            assert merge_intervals([[1, 3], [2, 4], [6, 8]]) == [[1, 4], [6, 8]]
            assert merge_intervals([]) == []


        def test_randomized():
            random.seed(1212)
            for _ in range(50):
                intervals = []
                for _ in range(random.randint(0, 6)):
                    a = random.randint(0, 10)
                    b = random.randint(a, a + random.randint(0, 5))
                    intervals.append([a, b])
                assert merge_intervals(intervals) == oracle(intervals)
        """
    ).strip()
    assert_tests = [
        "assert merge_intervals([[1, 3], [2, 4], [6, 8]]) == [[1, 4], [6, 8]]",
        "assert merge_intervals([]) == []",
    ]
    return _make_task(
        prompt,
        "merge_intervals",
        "(intervals)",
        reference_tests,
        assert_tests,
        category="algorithms",
        difficulty=3,
        tags=["heldout_only"],
    )


def _run_length_encode_task() -> dict[str, str | list[str]]:
    prompt = "Run-length encode a string into list of (char, count) pairs."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert run_length_encode("aaabb") == [("a", 3), ("b", 2)]
            assert run_length_encode("") == []
        """
    ).strip()
    assert_tests = [
        "assert run_length_encode('aaabb') == [('a', 3), ('b', 2)]",
        "assert run_length_encode('') == []",
    ]
    return _make_task(
        prompt, "run_length_encode", "(text)", reference_tests, assert_tests, category="strings", difficulty=2
    )


def _run_length_decode_task() -> dict[str, str | list[str]]:
    prompt = "Decode a run-length list of (char, count) pairs back into a string."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert run_length_decode([("a", 3), ("b", 2)]) == "aaabb"
            assert run_length_decode([]) == ""
        """
    ).strip()
    assert_tests = [
        "assert run_length_decode([('a', 3), ('b', 2)]) == 'aaabb'",
        "assert run_length_decode([]) == ''",
    ]
    return _make_task(
        prompt, "run_length_decode", "(pairs)", reference_tests, assert_tests, category="strings", difficulty=2
    )


def _sliding_window_max_sum_task() -> dict[str, str | list[str]]:
    prompt = "Return the maximum sum over all contiguous windows of size k."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values, k):
            if k <= 0 or k > len(values):
                return None
            return max(sum(values[i:i+k]) for i in range(len(values) - k + 1))


        def test_examples():
            assert max_window_sum([1, 2, 3, 4], 2) == 7
            assert max_window_sum([5], 1) == 5
            assert max_window_sum([1, 2], 3) is None


        def test_randomized():
            random.seed(1313)
            for _ in range(50):
                size = random.randint(1, 30)
                values = [random.randint(-5, 5) for _ in range(size)]
                k = random.randint(1, size)
                assert max_window_sum(values, k) == oracle(values, k)
        """
    ).strip()
    assert_tests = [
        "assert max_window_sum([1, 2, 3, 4], 2) == 7",
        "assert max_window_sum([1, 2], 3) is None",
    ]
    return _make_task(
        prompt, "max_window_sum", "(values, k)", reference_tests, assert_tests, category="lists", difficulty=2
    )


def _longest_common_prefix_task() -> dict[str, str | list[str]]:
    prompt = "Return the longest common prefix for a list of strings."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert longest_common_prefix(["flower", "flow", "flight"]) == "fl"
            assert longest_common_prefix(["dog", "racecar"]) == ""
            assert longest_common_prefix([]) == ""
        """
    ).strip()
    assert_tests = [
        "assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'",
        "assert longest_common_prefix([]) == ''",
    ]
    return _make_task(
        prompt, "longest_common_prefix", "(words)", reference_tests, assert_tests, category="algorithms", difficulty=1
    )


def _prefix_sums_task() -> dict[str, str | list[str]]:
    prompt = "Return prefix sums for a list of numbers."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert prefix_sums([1, 2, 3]) == [1, 3, 6]
            assert prefix_sums([]) == []
        """
    ).strip()
    assert_tests = [
        "assert prefix_sums([1, 2, 3]) == [1, 3, 6]",
        "assert prefix_sums([]) == []",
    ]
    return _make_task(prompt, "prefix_sums", "(values)", reference_tests, assert_tests, category="lists", difficulty=1)


def _chunk_list_task() -> dict[str, str | list[str]]:
    prompt = "Split a list into chunks of size n. The last chunk may be shorter."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
            assert chunk_list([], 3) == []
        """
    ).strip()
    assert_tests = [
        "assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]",
        "assert chunk_list([], 3) == []",
    ]
    return _make_task(prompt, "chunk_list", "(values, n)", reference_tests, assert_tests, category="lists", difficulty=1)


def _rotate_right_task() -> dict[str, str | list[str]]:
    prompt = "Rotate a list right by k positions."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]
            assert rotate_right([], 3) == []
        """
    ).strip()
    assert_tests = [
        "assert rotate_right([1, 2, 3, 4], 1) == [4, 1, 2, 3]",
        "assert rotate_right([], 3) == []",
    ]
    return _make_task(
        prompt, "rotate_right", "(values, k)", reference_tests, assert_tests, category="lists", difficulty=1
    )


def _k_smallest_task() -> dict[str, str | list[str]]:
    prompt = "Return the k smallest numbers from a list, sorted ascending."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def test_examples():
            assert k_smallest([5, 1, 3, 2], 2) == [1, 2]
            assert k_smallest([], 3) == []


        def test_randomized():
            random.seed(1414)
            for _ in range(40):
                size = random.randint(0, 20)
                values = [random.randint(-10, 10) for _ in range(size)]
                k = random.randint(0, size) if size else 0
                assert k_smallest(values, k) == sorted(values)[:k]
        """
    ).strip()
    assert_tests = [
        "assert k_smallest([5, 1, 3, 2], 2) == [1, 2]",
        "assert k_smallest([], 3) == []",
    ]
    return _make_task(prompt, "k_smallest", "(values, k)", reference_tests, assert_tests, category="lists", difficulty=2)


def _bugfix_is_sorted_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: is_sorted should return True for non-decreasing lists with equal neighbors."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert is_sorted([1, 2, 2, 3]) is True
            assert is_sorted([3, 2, 1]) is False
            assert is_sorted([]) is True
        """
    ).strip()
    assert_tests = [
        "assert is_sorted([1, 2, 2, 3]) is True",
        "assert is_sorted([3, 2, 1]) is False",
    ]
    return _make_task(
        prompt,
        "is_sorted",
        "(values)",
        reference_tests,
        assert_tests,
        category="lists",
        difficulty=2,
        tags=["heldout_only"],
    )


def _bugfix_count_words_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: count_words should ignore multiple spaces and tabs."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert count_words("hello   world") == 2
            assert count_words("  a\\t b\\n c ") == 3
        """
    ).strip()
    assert_tests = [
        "assert count_words('hello   world') == 2",
        "assert count_words('  a\\t b\\n c ') == 3",
    ]
    return _make_task(prompt, "count_words", "(text)", reference_tests, assert_tests, category="strings", difficulty=1)


def _bugfix_title_case_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: title_case should capitalize each word while preserving hyphenated parts."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert title_case("hello world") == "Hello World"
            assert title_case("state-of-the-art") == "State-Of-The-Art"
        """
    ).strip()
    assert_tests = [
        "assert title_case('hello world') == 'Hello World'",
        "assert title_case('state-of-the-art') == 'State-Of-The-Art'",
    ]
    return _make_task(prompt, "title_case", "(text)", reference_tests, assert_tests, category="strings", difficulty=2)


def _bugfix_unique_chars_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: unique_chars should treat uppercase and lowercase as the same."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert unique_chars("AaBb") == ["a", "b"]
            assert unique_chars("") == []
        """
    ).strip()
    assert_tests = [
        "assert unique_chars('AaBb') == ['a', 'b']",
        "assert unique_chars('') == []",
    ]
    return _make_task(
        prompt,
        "unique_chars",
        "(text)",
        reference_tests,
        assert_tests,
        category="strings",
        difficulty=2,
        tags=["heldout_only"],
    )


def _bugfix_balance_parens_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: balance_parens should return True for empty input and ignore non-paren chars."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert balance_parens("(a)b") is True
            assert balance_parens("(()") is False
            assert balance_parens("") is True
        """
    ).strip()
    assert_tests = [
        "assert balance_parens('(a)b') is True",
        "assert balance_parens('(()') is False",
    ]
    return _make_task(
        prompt,
        "balance_parens",
        "(text)",
        reference_tests,
        assert_tests,
        category="algorithms",
        difficulty=2,
        tags=["heldout_only"],
    )


def _bugfix_median_task() -> dict[str, str | list[str]]:
    prompt = "Fix the bug: median should average the two middle values for even-length lists."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert median([1, 2, 3]) == 2
            assert median([1, 2, 3, 4]) == 2.5
        """
    ).strip()
    assert_tests = [
        "assert median([1, 2, 3]) == 2",
        "assert median([1, 2, 3, 4]) == 2.5",
    ]
    return _make_task(
        prompt,
        "median",
        "(values)",
        reference_tests,
        assert_tests,
        category="math",
        difficulty=2,
        tags=["heldout_only"],
    )


def _max_pair_sum_task() -> dict[str, str | list[str]]:
    prompt = "Return the maximum sum of any two numbers in the list."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values):
            if len(values) < 2:
                return None
            best = None
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    total = values[i] + values[j]
                    if best is None or total > best:
                        best = total
            return best


        def test_examples():
            assert max_pair_sum([1, 2, 3]) == 5
            assert max_pair_sum([5]) is None


        def test_randomized():
            random.seed(1515)
            for _ in range(40):
                size = random.randint(1, 60)
                values = [random.randint(-50, 50) for _ in range(size)]
                assert max_pair_sum(values) == oracle(values)
        """
    ).strip()
    assert_tests = [
        "assert max_pair_sum([1, 2, 3]) == 5",
        "assert max_pair_sum([5]) is None",
    ]
    return _make_task(
        prompt, "max_pair_sum", "(values)", reference_tests, assert_tests, category="lists", difficulty=2
    )


def _first_duplicate_task() -> dict[str, str | list[str]]:
    prompt = "Return the first duplicated value in order of appearance, or None if all unique."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert first_duplicate([1, 2, 3, 2, 1]) == 2
            assert first_duplicate([1, 2, 3]) is None
        """
    ).strip()
    assert_tests = [
        "assert first_duplicate([1, 2, 3, 2, 1]) == 2",
        "assert first_duplicate([1, 2, 3]) is None",
    ]
    return _make_task(
        prompt, "first_duplicate", "(values)", reference_tests, assert_tests, category="lists", difficulty=2
    )


def _min_subarray_len_task() -> dict[str, str | list[str]]:
    prompt = "Return the minimum length of a contiguous subarray with sum >= target, or 0 if none."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def oracle(values, target):
            best = None
            for i in range(len(values)):
                total = 0
                for j in range(i, len(values)):
                    total += values[j]
                    if total >= target:
                        length = j - i + 1
                        if best is None or length < best:
                            best = length
                        break
            return best or 0


        def test_examples():
            assert min_subarray_len([2, 3, 1, 2, 4, 3], 7) == 2
            assert min_subarray_len([1, 1, 1], 5) == 0


        def test_randomized():
            random.seed(1616)
            for _ in range(30):
                size = random.randint(1, 40)
                values = [random.randint(1, 6) for _ in range(size)]
                target = random.randint(1, 20)
                assert min_subarray_len(values, target) == oracle(values, target)
        """
    ).strip()
    assert_tests = [
        "assert min_subarray_len([2, 3, 1, 2, 4, 3], 7) == 2",
        "assert min_subarray_len([1, 1, 1], 5) == 0",
    ]
    return _make_task(
        prompt,
        "min_subarray_len",
        "(values, target)",
        reference_tests,
        assert_tests,
        category="algorithms",
        difficulty=3,
    )


def _longest_run_task() -> dict[str, str | list[str]]:
    prompt = "Return the length of the longest run of identical values."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert longest_run([1, 1, 2, 2, 2, 3]) == 3
            assert longest_run([]) == 0
        """
    ).strip()
    assert_tests = [
        "assert longest_run([1, 1, 2, 2, 2, 3]) == 3",
        "assert longest_run([]) == 0",
    ]
    return _make_task(
        prompt, "longest_run", "(values)", reference_tests, assert_tests, category="lists", difficulty=1
    )


def _find_peak_task() -> dict[str, str | list[str]]:
    prompt = "Return any peak element (>= neighbors) from a list, or None if empty."
    reference_tests = textwrap.dedent(
        """
        import pytest
        import random
        from solution import *


        def test_examples():
            assert find_peak([1, 3, 2]) in {3}
            assert find_peak([]) is None


        def test_randomized():
            random.seed(1717)
            for _ in range(40):
                size = random.randint(1, 30)
                values = [random.randint(-5, 5) for _ in range(size)]
                peak = find_peak(values)
                assert peak in values
                idx = values.index(peak)
                left_ok = idx == 0 or values[idx] >= values[idx - 1]
                right_ok = idx == len(values) - 1 or values[idx] >= values[idx + 1]
                assert left_ok and right_ok
        """
    ).strip()
    assert_tests = [
        "assert find_peak([1, 3, 2]) in {3}",
        "assert find_peak([]) is None",
    ]
    return _make_task(
        prompt, "find_peak", "(values)", reference_tests, assert_tests, category="algorithms", difficulty=2
    )


def _window_distinct_count_task() -> dict[str, str | list[str]]:
    prompt = "Count distinct values in each window of size k."
    reference_tests = textwrap.dedent(
        """
        import pytest
        from solution import *


        def test_examples():
            assert window_distinct_counts([1, 2, 1, 3, 2], 3) == [2, 3, 3]
            assert window_distinct_counts([1, 1, 1], 2) == [1, 1]
        """
    ).strip()
    assert_tests = [
        "assert window_distinct_counts([1, 2, 1, 3, 2], 3) == [2, 3, 3]",
        "assert window_distinct_counts([1, 1, 1], 2) == [1, 1]",
    ]
    return _make_task(
        prompt,
        "window_distinct_counts",
        "(values, k)",
        reference_tests,
        assert_tests,
        category="lists",
        difficulty=2,
    )


def _interleave_range(start: int, stop: int) -> list[int]:
    values = []
    for k in range(start, stop + 1):
        values.append(k)
        values.append(-k)
    return values


def generate_tasks(count: int) -> list[dict[str, str | list[str]]]:
    core_tasks = [
        _rotate_left_task(),
        _is_palindrome_alnum_task(),
        _two_sum_indices_task(),
        _clamp_task(),
        _gcd_task(),
        _lcm_task(),
        _piecewise_task(),
        _dedupe_preserve_task(),
        _parse_int_list_task(),
        _flatten_one_level_task(),
        _max_subarray_sum_task(),
        _bfs_distance_task(),
        _parse_date_iso_task(),
        _invert_mapping_task(),
        _merge_with_precedence_task(),
        _parse_key_value_task(),
        _parse_csv_row_task(),
        _parse_csv_quoted_task(),
        _parse_env_block_task(),
        _parse_range_list_task(),
        _parse_bool_tokens_task(),
        _parse_int_matrix_task(),
        _normalize_whitespace_task(),
        _parse_duration_minutes_task(),
        _parse_scored_pairs_task(),
        _merge_intervals_task(),
        _run_length_encode_task(),
        _run_length_decode_task(),
        _sliding_window_max_sum_task(),
        _longest_common_prefix_task(),
        _prefix_sums_task(),
        _chunk_list_task(),
        _rotate_right_task(),
        _k_smallest_task(),
        _bugfix_is_sorted_task(),
        _bugfix_count_words_task(),
        _bugfix_title_case_task(),
        _bugfix_unique_chars_task(),
        _bugfix_balance_parens_task(),
        _bugfix_median_task(),
        _max_pair_sum_task(),
        _first_duplicate_task(),
        _min_subarray_len_task(),
        _longest_run_task(),
        _find_peak_task(),
        _window_distinct_count_task(),
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


def write_tasks(tasks: list[dict[str, str | list[str]]], output_path: Path) -> None:
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
