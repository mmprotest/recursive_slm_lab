from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .base import LLMBackend, LLMResponse


@dataclass
class MockBackend(LLMBackend):
    model_name: str = "mock-model"
    seed: int = 1337
    baseline_success_rate: float = 0.35

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> LLMResponse:
        _ = max_tokens, temperature, top_p, top_k
        prompt = ""
        for message in messages:
            if message.get("role") == "user":
                prompt = message.get("content", "")
        code = self._generate_code(prompt)
        return LLMResponse(text=code, model=self.model_name)

    def _generate_code(self, prompt: str) -> str:
        function_name = self._extract_function_name(prompt)
        signature = self._extract_signature(prompt)

        example_code = self._extract_example_code(prompt)
        memory_code = self._extract_memory_context_code(prompt)

        if example_code:
            adapted = self._adapt_example(example_code, function_name, signature, prompt)
            return adapted or self._rename_example(example_code, function_name, signature)

        if memory_code:
            adapted = self._adapt_example(memory_code, function_name, signature, prompt)
            return adapted or self._rename_example(memory_code, function_name, signature)

        if self._has_learned_signal(prompt):
            synthesized = self._synthesize_solution(function_name, signature, prompt)
            if synthesized:
                return synthesized

        if self._should_attempt(function_name):
            synthesized = self._synthesize_solution(function_name, signature, prompt)
            if synthesized:
                return synthesized

        return f"def {function_name}{signature}:\n    return None\n"

    def _should_attempt(self, function_name: str) -> bool:
        digest = hashlib.sha256(f"{self.seed}:{function_name}".encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 100
        return bucket < int(self.baseline_success_rate * 100)

    def _adapt_example(self, example_code: str, function_name: str, signature: str, prompt: str) -> str | None:
        if "add_const" in function_name and re.search(r"return\s+n\s*\+\s*\d+", example_code):
            value = self._parse_const_suffix(function_name)
            return f"def {function_name}{signature}:\n    return n + {value}\n"
        if "mul_const" in function_name and re.search(r"return\s+n\s*\*\s*\d+", example_code):
            value = self._parse_const_suffix(function_name)
            return f"def {function_name}{signature}:\n    return n * {value}\n"
        if "is_divisible_by" in function_name and re.search(r"return\s+n\s*%\s*\d+\s*==\s*0", example_code):
            value = int(function_name.split("_")[-1])
            return f"def {function_name}{signature}:\n    return n % {value} == 0\n"
        if "power_" in function_name and re.search(r"return\s+n\s*\*\*\s*\d+", example_code):
            value = int(function_name.split("_")[-1])
            return f"def {function_name}{signature}:\n    return n ** {value}\n"
        example_name = self._extract_def_name(example_code)
        if example_name and example_name == function_name:
            return self._rename_example(example_code, function_name, signature)
        return None

    @staticmethod
    def _rename_example(example_code: str, function_name: str, signature: str) -> str:
        lines = example_code.strip().splitlines()
        if not lines:
            return f"def {function_name}{signature}:\n    return None\n"
        if lines[0].startswith("def "):
            lines[0] = f"def {function_name}{signature}:"
        return "\n".join(lines).rstrip() + "\n"

    def _synthesize_solution(self, function_name: str, signature: str, prompt: str) -> str | None:
        if "add_const" in function_name:
            value = self._parse_const_suffix(function_name)
            return f"def {function_name}{signature}:\n    return n + {value}\n"
        if "mul_const" in function_name:
            value = self._parse_const_suffix(function_name)
            return f"def {function_name}{signature}:\n    return n * {value}\n"
        if "is_divisible_by" in function_name:
            value = int(function_name.split("_")[-1])
            return f"def {function_name}{signature}:\n    return n % {value} == 0\n"
        if "power_" in function_name:
            value = int(function_name.split("_")[-1])
            return f"def {function_name}{signature}:\n    return n ** {value}\n"
        if function_name == "is_even":
            return f"def {function_name}{signature}:\n    return n % 2 == 0\n"
        if function_name == "is_odd":
            return f"def {function_name}{signature}:\n    return n % 2 == 1\n"
        if function_name == "reverse_string":
            return f"def {function_name}{signature}:\n    return text[::-1]\n"
        if function_name == "count_vowels":
            return f"def {function_name}{signature}:\n    return sum(1 for ch in text.lower() if ch in 'aeiou')\n"
        if function_name == "list_sum":
            return f"def {function_name}{signature}:\n    return sum(values)\n"
        if function_name == "list_max":
            return f"def {function_name}{signature}:\n    return max(values) if values else None\n"
        if function_name == "list_min":
            return f"def {function_name}{signature}:\n    return min(values) if values else None\n"
        if function_name == "list_avg":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(values) / len(values) if values else None\n"
            )
        if function_name == "list_len":
            return f"def {function_name}{signature}:\n    return len(values)\n"
        if function_name == "list_reverse":
            return f"def {function_name}{signature}:\n    return list(reversed(values))\n"
        if function_name == "list_sort":
            return f"def {function_name}{signature}:\n    return sorted(values)\n"
        if function_name == "list_unique":
            return (
                f"def {function_name}{signature}:\n"
                "    seen = []\n"
                "    for item in values:\n"
                "        if item not in seen:\n"
                "            seen.append(item)\n"
                "    return seen\n"
            )
        if function_name == "factorial":
            return (
                f"def {function_name}{signature}:\n"
                "    if n < 0:\n"
                "        return None\n"
                "    result = 1\n"
                "    for i in range(2, n + 1):\n"
                "        result *= i\n"
                "    return result\n"
            )
        if function_name == "fibonacci":
            return (
                f"def {function_name}{signature}:\n"
                "    if n < 0:\n"
                "        return None\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            )
        if function_name == "is_palindrome":
            return (
                f"def {function_name}{signature}:\n"
                "    cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return cleaned == cleaned[::-1]\n"
            )
        if function_name == "to_upper":
            return f"def {function_name}{signature}:\n    return text.upper()\n"
        if function_name == "to_lower":
            return f"def {function_name}{signature}:\n    return text.lower()\n"
        if function_name == "count_words":
            return (
                f"def {function_name}{signature}:\n"
                "    return len(text.split())\n"
            )
        if function_name == "strip_non_alnum":
            return (
                f"def {function_name}{signature}:\n"
                "    return ''.join(ch for ch in text if ch.isalnum())\n"
            )
        if function_name == "is_prime":
            return (
                f"def {function_name}{signature}:\n"
                "    if n < 2:\n"
                "        return False\n"
                "    i = 2\n"
                "    while i * i <= n:\n"
                "        if n % i == 0:\n"
                "            return False\n"
                "        i += 1\n"
                "    return True\n"
            )
        if function_name == "gcd":
            return (
                f"def {function_name}{signature}:\n"
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return abs(a)\n"
            )
        if function_name == "lcm":
            return (
                f"def {function_name}{signature}:\n"
                "    if a == 0 or b == 0:\n"
                "        return 0\n"
                "    x, y = a, b\n"
                "    while y:\n"
                "        x, y = y, x % y\n"
                "    return abs(a * b) // abs(x)\n"
            )
        if function_name == "clamp":
            return (
                f"def {function_name}{signature}:\n"
                "    return max(low, min(high, x))\n"
            )
        if function_name == "median":
            return (
                f"def {function_name}{signature}:\n"
                "    if not values:\n"
                "        return None\n"
                "    sorted_vals = sorted(values)\n"
                "    mid = len(values) // 2\n"
                "    if len(values) % 2 == 1:\n"
                "        return sorted_vals[mid]\n"
                "    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2\n"
            )
        if function_name == "count_unique":
            return (
                f"def {function_name}{signature}:\n"
                "    return len(set(values))\n"
            )
        if function_name == "remove_negatives":
            return (
                f"def {function_name}{signature}:\n"
                "    return [value for value in values if value >= 0]\n"
            )
        if function_name == "sum_even":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(value for value in values if value % 2 == 0)\n"
            )
        if function_name == "count_uppercase":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(1 for ch in text if ch.isupper())\n"
            )
        if function_name == "flatten":
            return (
                f"def {function_name}{signature}:\n"
                "    result = []\n"
                "    for row in matrix:\n"
                "        result.extend(row)\n"
                "    return result\n"
            )
        if function_name == "pairwise_sum":
            return (
                f"def {function_name}{signature}:\n"
                "    return [a + b for a, b in zip(values[::2], values[1::2])]\n"
            )
        if function_name == "rotate_left":
            return (
                f"def {function_name}{signature}:\n"
                "    if not values:\n"
                "        return []\n"
                "    return values[1:] + values[:1]\n"
            )
        if function_name == "is_palindrome_alnum":
            return (
                f"def {function_name}{signature}:\n"
                "    filtered = ''.join(ch.lower() for ch in text if ch.isalnum())\n"
                "    return filtered == filtered[::-1]\n"
            )
        if function_name == "two_sum_indices":
            return (
                f"def {function_name}{signature}:\n"
                "    for i in range(len(nums)):\n"
                "        for j in range(i + 1, len(nums)):\n"
                "            if nums[i] + nums[j] == target:\n"
                "                return (i, j)\n"
                "    return (-1, -1)\n"
            )
        if function_name == "dedupe_preserve":
            return (
                f"def {function_name}{signature}:\n"
                "    seen = set()\n"
                "    result = []\n"
                "    for item in values:\n"
                "        if item in seen:\n"
                "            continue\n"
                "        seen.add(item)\n"
                "        result.append(item)\n"
                "    return result\n"
            )
        if function_name == "parse_int_list":
            return (
                f"def {function_name}{signature}:\n"
                "    if not text.strip():\n"
                "        return []\n"
                "    normalized = text.replace(',', ' ')\n"
                "    parts = [p for p in normalized.split() if p]\n"
                "    return [int(p) for p in parts]\n"
            )
        if function_name == "max_subarray_sum":
            return (
                f"def {function_name}{signature}:\n"
                "    if not values:\n"
                "        return 0\n"
                "    best = None\n"
                "    for i in range(len(values)):\n"
                "        total = 0\n"
                "        for j in range(i, len(values)):\n"
                "            total += values[j]\n"
                "            if best is None or total > best:\n"
                "                best = total\n"
                "    return best if best is not None else 0\n"
            )
        if function_name == "bfs_distance":
            return (
                f"def {function_name}{signature}:\n"
                "    from collections import deque\n"
                "    if start == goal:\n"
                "        return 0\n"
                "    visited = {start}\n"
                "    queue = deque([(start, 0)])\n"
                "    while queue:\n"
                "        node, dist = queue.popleft()\n"
                "        for neighbor in graph.get(node, []):\n"
                "            if neighbor in visited:\n"
                "                continue\n"
                "            if neighbor == goal:\n"
                "                return dist + 1\n"
                "            visited.add(neighbor)\n"
                "            queue.append((neighbor, dist + 1))\n"
                "    return -1\n"
            )
        return None

    @staticmethod
    def _extract_function_name(prompt: str) -> str:
        match = re.search(r"Function Name:\s*(\w+)", prompt)
        return match.group(1) if match else "solution"

    @staticmethod
    def _extract_signature(prompt: str) -> str:
        match = re.search(r"Signature:\s*(\([^\)]*\))", prompt)
        return match.group(1) if match else "()"

    @staticmethod
    def _extract_example_code(prompt: str) -> str | None:
        match = re.search(r"EXAMPLE_CODE_START\n(.*?)\nEXAMPLE_CODE_END", prompt, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_memory_context_code(prompt: str) -> str | None:
        match = re.search(r"Memory Context:\n(.*)", prompt, re.DOTALL)
        if not match:
            return None
        lines = match.group(1).splitlines()
        for line in lines:
            if line.strip().startswith("def "):
                return line + "\n" + "\n".join(lines[lines.index(line) + 1 :])
        return None

    @staticmethod
    def _extract_def_name(code: str) -> str | None:
        match = re.search(r"def\s+(\w+)\s*\(", code)
        return match.group(1) if match else None

    @staticmethod
    def _parse_const_suffix(function_name: str) -> int:
        suffix = function_name.split("_")[-1]
        if suffix.startswith("neg"):
            return -int(suffix.replace("neg", ""))
        return int(suffix)

    @staticmethod
    def _has_learned_signal(prompt: str) -> bool:
        return "Memory Context:" in prompt
