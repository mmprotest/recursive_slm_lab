from __future__ import annotations

CODE_ONLY_CONTRACT = (
    "Return ONLY valid Python code. Define ONLY the requested function. "
    "Do not include tests, explanations, or markdown."
)


def build_prompt(
    task_prompt: str,
    function_name: str,
    signature: str,
    memory_context: str | None,
    example_code: str | None,
) -> str:
    memory_block = ""
    if memory_context:
        memory_block = (
            "\n\nMemory Context:\n"
            "Use the memory snippets below if relevant.\n"
            f"{memory_context}\n"
        )
    if example_code:
        memory_block += (
            "\nEXAMPLE_CODE_START\n"
            f"{example_code}\n"
            "EXAMPLE_CODE_END\n"
        )
    return (
        f"{CODE_ONLY_CONTRACT}\n\n"
        f"Task: {task_prompt}\n"
        f"Function Name: {function_name}\n"
        f"Signature: {signature}\n"
        f"Implement the function accordingly."
        f"{memory_block}"
    )
