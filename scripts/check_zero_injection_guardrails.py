#!/usr/bin/env python3
"""
Static guardrails for SafeClaw zero-injection invariants.

Fails CI if critical safety-context patterns regress.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
CLAW_AGENT = ROOT / "med_safety_gym" / "claw_agent.py"


def fail(msg: str) -> None:
    print(f"[guardrail] FAIL: {msg}")
    sys.exit(1)


def check_text_patterns(src: str) -> None:
    # Raw user prompt/history must never be concatenated into parity context checks.
    banned_tokens = ("USER PROMPT:", "HISTORY:")
    for token in banned_tokens:
        if token in src:
            fail(f"Found banned parity-taint token in claw_agent.py: {token}")


def check_ast_contract(src: str) -> None:
    tree = ast.parse(src)

    class ContractVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.saw_verified_context_fallback = False
            self.saw_output_context_assignment = False
            self.apply_safety_with_safe_context = 0
            self.apply_safety_with_context = 0

        def visit_Assign(self, node: ast.Assign) -> None:
            # verified_context = parity_context if parity_context is not None else context
            if any(isinstance(t, ast.Name) and t.id == "verified_context" for t in node.targets):
                if (
                    isinstance(node.value, ast.IfExp)
                    and isinstance(node.value.body, ast.Name)
                    and node.value.body.id == "parity_context"
                    and isinstance(node.value.orelse, ast.Name)
                    and node.value.orelse.id == "context"
                ):
                    self.saw_verified_context_fallback = True

            # output_context = verified_context
            if any(isinstance(t, ast.Name) and t.id == "output_context" for t in node.targets):
                if isinstance(node.value, ast.Name) and node.value.id == "verified_context":
                    self.saw_output_context_assignment = True

            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # _apply_safety_gate(..., safe_context, ...)
            if isinstance(node.func, ast.Attribute) and node.func.attr == "_apply_safety_gate":
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Name):
                    arg_name = node.args[1].id
                    if arg_name in ("verified_context", "output_context"):
                        self.apply_safety_with_safe_context += 1
                    if arg_name == "context":
                        self.apply_safety_with_context += 1

            self.generic_visit(node)

    visitor = ContractVisitor()
    visitor.visit(tree)

    if not visitor.saw_verified_context_fallback:
        fail("Missing verified_context fallback assignment from parity_context/context.")
    if not visitor.saw_output_context_assignment:
        fail("Missing output_context = verified_context assignment.")
    if visitor.apply_safety_with_safe_context < 2:
        fail("Expected both pre- and post-generation _apply_safety_gate calls with safe context.")
    if visitor.apply_safety_with_context > 0:
        fail("Found unsafe _apply_safety_gate(..., context, ...) call.")


def main() -> None:
    src = CLAW_AGENT.read_text(encoding="utf-8")
    check_text_patterns(src)
    check_ast_contract(src)
    print("[guardrail] PASS: zero-injection guardrails intact.")


if __name__ == "__main__":
    main()
