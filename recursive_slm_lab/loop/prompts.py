from __future__ import annotations

from ..policy import DEFAULT_POLICY, Policy


def build_prompt(
    task_prompt: str,
    function_name: str,
    signature: str,
    memory_context: str | None,
    example_code: str | None,
    policy: Policy | None = None,
) -> str:
    policy = policy or DEFAULT_POLICY
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
    return policy.prompt_template.format(
        prompt_contract=policy.prompt_contract,
        task_prompt=task_prompt,
        function_name=function_name,
        signature=signature,
        memory_blocks=memory_block,
    )
