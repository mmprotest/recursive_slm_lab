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

    def generate(self, prompt: str, max_tokens: int, temperature: float) -> LLMResponse:
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
                "    mid = len(sorted_vals) // 2\n"
                "    if len(sorted_vals) % 2 == 1:\n"
                "        return sorted_vals[mid]\n"
                "    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2\n"
            )
        if function_name == "mode":
            return (
                f"def {function_name}{signature}:\n"
                "    counts = {}\n"
                "    for item in values:\n"
                "        counts[item] = counts.get(item, 0) + 1\n"
                "    max_count = max(counts.values())\n"
                "    candidates = [k for k, v in counts.items() if v == max_count]\n"
                "    return min(candidates)\n"
            )
        if function_name == "flatten_list":
            return (
                f"def {function_name}{signature}:\n"
                "    return [item for sub in values for item in sub]\n"
            )
        if function_name == "pairwise_sum":
            return (
                f"def {function_name}{signature}:\n"
                "    return [values[i] + values[i + 1] for i in range(len(values) - 1)]\n"
            )
        if function_name == "even_numbers":
            return (
                f"def {function_name}{signature}:\n"
                "    return [v for v in values if v % 2 == 0]\n"
            )
        if function_name == "odd_numbers":
            return (
                f"def {function_name}{signature}:\n"
                "    return [v for v in values if v % 2 == 1]\n"
            )
        if function_name == "remove_negatives":
            return (
                f"def {function_name}{signature}:\n"
                "    return [v for v in values if v >= 0]\n"
            )
        if function_name == "dict_keys_sorted":
            return (
                f"def {function_name}{signature}:\n"
                "    return sorted(mapping.keys())\n"
            )
        if function_name == "dict_values_sum":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(mapping.values())\n"
            )
        if function_name == "invert_dict":
            return (
                f"def {function_name}{signature}:\n"
                "    return {v: k for k, v in mapping.items()}\n"
            )
        if function_name == "merge_dicts":
            return (
                f"def {function_name}{signature}:\n"
                "    merged = dict(a)\n"
                "    merged.update(b)\n"
                "    return merged\n"
            )
        if function_name == "count_char":
            return (
                f"def {function_name}{signature}:\n"
                "    return text.count(char)\n"
            )
        if function_name == "starts_with_vowel":
            return (
                f"def {function_name}{signature}:\n"
                "    return bool(text) and text[0].lower() in 'aeiou'\n"
            )
        if function_name == "ends_with_vowel":
            return (
                f"def {function_name}{signature}:\n"
                "    return bool(text) and text[-1].lower() in 'aeiou'\n"
            )
        if function_name == "safe_divide":
            return (
                f"def {function_name}{signature}:\n"
                "    return None if b == 0 else a / b\n"
            )
        if function_name == "sign":
            return (
                f"def {function_name}{signature}:\n"
                "    if x > 0:\n"
                "        return 1\n"
                "    if x < 0:\n"
                "        return -1\n"
                "    return 0\n"
            )
        if function_name == "absolute":
            return (
                f"def {function_name}{signature}:\n"
                "    return abs(x)\n"
            )
        if function_name == "modulo_abs":
            return (
                f"def {function_name}{signature}:\n"
                "    return abs(a) % b\n"
            )
        if function_name == "repeat_string":
            return (
                f"def {function_name}{signature}:\n"
                "    return text * n\n"
            )
        if function_name == "join_with_comma":
            return (
                f"def {function_name}{signature}:\n"
                "    return ','.join(values)\n"
            )
        if function_name == "nth_char":
            return (
                f"def {function_name}{signature}:\n"
                "    return text[n] if 0 <= n < len(text) else ''\n"
            )
        if function_name == "first_last":
            return (
                f"def {function_name}{signature}:\n"
                "    if not text:\n"
                "        return ('', '')\n"
                "    return (text[0], text[-1])\n"
            )
        if function_name == "sum_of_squares":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(v * v for v in values)\n"
            )
        if function_name == "sum_range":
            return (
                f"def {function_name}{signature}:\n"
                "    if start > end:\n"
                "        return 0\n"
                "    return sum(range(start, end + 1))\n"
            )
        if function_name == "range_list":
            return (
                f"def {function_name}{signature}:\n"
                "    if start > end:\n"
                "        return []\n"
                "    return list(range(start, end + 1))\n"
            )
        if function_name == "contains_value":
            return (
                f"def {function_name}{signature}:\n"
                "    return value in values\n"
            )
        if function_name == "index_of":
            return (
                f"def {function_name}{signature}:\n"
                "    return values.index(value) if value in values else -1\n"
            )
        if function_name == "replace_spaces":
            return (
                f"def {function_name}{signature}:\n"
                "    return text.replace(' ', '_')\n"
            )
        if function_name == "count_true":
            return (
                f"def {function_name}{signature}:\n"
                "    return sum(1 for v in values if v)\n"
            )
        if function_name == "all_positive":
            return (
                f"def {function_name}{signature}:\n"
                "    return all(v > 0 for v in values)\n"
            )
        if function_name == "any_negative":
            return (
                f"def {function_name}{signature}:\n"
                "    return any(v < 0 for v in values)\n"
            )
        if function_name == "clamp_list":
            return (
                f"def {function_name}{signature}:\n"
                "    return [max(low, min(high, v)) for v in values]\n"
            )
        if function_name == "rotate_left":
            return (
                f"def {function_name}{signature}:\n"
                "    return values[1:] + values[:1] if values else []\n"
            )
        if function_name == "rotate_right":
            return (
                f"def {function_name}{signature}:\n"
                "    return values[-1:] + values[:-1] if values else []\n"
            )
        if function_name == "triangle_area":
            return (
                f"def {function_name}{signature}:\n"
                "    return base * height / 2\n"
            )
        if function_name == "rectangle_area":
            return (
                f"def {function_name}{signature}:\n"
                "    return width * height\n"
            )
        if function_name == "circle_area":
            return (
                f"def {function_name}{signature}:\n"
                "    return 3.141592653589793 * r * r\n"
            )
        if function_name == "celsius_to_f":
            return (
                f"def {function_name}{signature}:\n"
                "    return (c * 9 / 5) + 32\n"
            )
        if function_name == "f_to_c":
            return (
                f"def {function_name}{signature}:\n"
                "    return (f - 32) * 5 / 9\n"
            )
        if function_name == "km_to_miles":
            return (
                f"def {function_name}{signature}:\n"
                "    return km * 0.621371\n"
            )
        if function_name == "miles_to_km":
            return (
                f"def {function_name}{signature}:\n"
                "    return miles * 1.60934\n"
            )
        if function_name == "count_digits":
            return (
                f"def {function_name}{signature}:\n"
                "    return len(str(abs(n)))\n"
            )

        return None

    @staticmethod
    def _extract_def_name(code: str) -> str | None:
        match = re.search(r"def\s+(\w+)\(", code)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _has_learned_signal(prompt: str) -> bool:
        hints = (
            "normalize case and remove non-alphanumeric",
            "vowel counting",
            "iterative accumulation",
            "factorial can be computed",
            "trial division",
            "median can be found",
            "mode can be computed",
            "preserve order by tracking seen",
            "use python built-ins like sum(values)",
            "use slicing [::-1] or reversed()",
        )
        lowered = prompt.lower()
        return "memory context" in lowered and any(hint in lowered for hint in hints)

    @staticmethod
    def _extract_function_name(prompt: str) -> str:
        match = re.search(r"Function Name:\s*(\w+)", prompt)
        if match:
            return match.group(1)
        match = re.search(r"def (\w+)\(", prompt)
        if match:
            return match.group(1)
        return "solution"

    @staticmethod
    def _extract_signature(prompt: str) -> str:
        match = re.search(r"Signature:\s*(\(.*\))", prompt)
        if match:
            return match.group(1)
        match = re.search(r"def \w+(\(.*\))", prompt)
        if match:
            return match.group(1)
        return "()"

    @staticmethod
    def _parse_const_suffix(function_name: str) -> int:
        suffix = function_name.split("_")[-1]
        if suffix.startswith("neg") and suffix[3:].isdigit():
            return -int(suffix[3:])
        return int(suffix)

    @staticmethod
    def _extract_example_code(prompt: str) -> str | None:
        marker = "EXAMPLE_CODE_START"
        if marker not in prompt:
            return None
        chunks = prompt.split(marker, 1)[1]
        end_marker = "EXAMPLE_CODE_END"
        if end_marker in chunks:
            code = chunks.split(end_marker, 1)[0]
        else:
            code = chunks
        code = code.strip()
        if not code.startswith("def"):
            return None
        return code + "\n"

    @staticmethod
    def _extract_memory_context_code(prompt: str) -> str | None:
        if "Memory Context:" not in prompt:
            return None
        snippet = prompt.split("Memory Context:", 1)[1]
        match = re.search(r"(def[\s\S]+)", snippet)
        if not match:
            return None
        code = match.group(1)
        for stopper in ("\n- [", "\nEXAMPLE_CODE_START"):
            if stopper in code:
                code = code.split(stopper, 1)[0]
        code = code.strip()
        if not code.startswith("def"):
            return None
        return code + "\n"
