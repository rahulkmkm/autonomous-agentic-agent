"""
Autonomous Agentic Agent (ReAct: Reason + Act)
==============================================

The agent receives a goal and repeats a loop until the goal is satisfied:

    Thought      -> the planner decides what to do next
    Action       -> the chosen tool is executed
    Observation  -> the tool's result is recorded in memory and fed back to the planner

Components
    Tool ABC + ToolRegistry   : calculator, web_search (mock), file_writer
    Planner                   : RuleBasedPlanner (offline, deterministic)
                                LLMPlanner (optional, uses the Anthropic API, falls back to rules)
    AutonomousAgent           : the ReAct orchestrator (loop, memory, step limit, error handling)

Usage
    python autonomous_agent.py                      # run the demo goals
    python autonomous_agent.py "calculate 12 * 7"   # run one goal
    python autonomous_agent.py -i                   # interactive mode
    AGENT_PLANNER=llm python autonomous_agent.py -i # LLM planner (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# ==========================================================================
# 1. TOOLS - the agent's capabilities
# ==========================================================================


@dataclass
class ToolResult:
    ok: bool
    output: str


class Tool(ABC):
    name: str
    description: str

    @abstractmethod
    def run(self, tool_input: str) -> ToolResult:
        """Execute the tool. Must never raise: return ToolResult(ok=False, ...) instead."""


# ---- Calculator: AST-based, no eval() -------------------------------------

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MAX_EXPONENT = 100


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"exponent too large (max {_MAX_EXPONENT})")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported syntax (only numbers and + - * / // % ** ( ) are allowed)")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluates an arithmetic expression, e.g. '25 * 4 + 10'. Supports + - * / // % ** ( )."

    def run(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression.replace("^", "**").strip(), mode="eval")
            result = _safe_eval(tree)
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            return ToolResult(True, str(result))
        except ZeroDivisionError:
            return ToolResult(False, "Error: division by zero")
        except (ValueError, SyntaxError, OverflowError) as exc:
            return ToolResult(False, f"Error evaluating '{expression}': {exc}")


# ---- Mock web search ---------------------------------------------------------


class MockSearchTool(Tool):
    name = "web_search"
    description = "Searches a small offline knowledge base (mock) for recent data or general knowledge."

    KNOWLEDGE_BASE = {
        "ai agent trends 2026": (
            "Autonomous multi-agent systems, self-correction loops, and tool-augmented "
            "LLMs are dominant trends in 2026."
        ),
        "python version": "Python 3.12+ is widely adopted for modern AI agent development.",
        "react framework": "ReAct interleaves reasoning traces with tool actions so an agent can plan and adapt.",
    }

    @staticmethod
    def _tokens(text: str) -> set:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def run(self, query: str) -> ToolResult:
        q = self._tokens(query)
        best_key, best_score = None, 0.0
        for key in self.KNOWLEDGE_BASE:
            k = self._tokens(key)
            score = len(k & q) / len(k)
            if score > best_score:
                best_key, best_score = key, score
        if best_key and best_score >= 0.6:
            return ToolResult(True, self.KNOWLEDGE_BASE[best_key])
        return ToolResult(False, f"No results found for '{query}' in the offline knowledge base.")


# ---- File writer (sandboxed) -----------------------------------------------


class FileWriterTool(Tool):
    name = "file_writer"
    description = "Saves text to a file inside the output directory. Input: JSON with 'filename' and 'content'."
    ALLOWED_SUFFIXES = {".txt", ".md", ".json", ".csv"}

    def __init__(self, output_dir: str = "agent_output"):
        self.output_dir = Path(output_dir).resolve()

    def run(self, data_json: str) -> ToolResult:
        try:
            data = json.loads(data_json)
            # Path(...).name drops any directory part, so '../../etc/x' cannot escape the sandbox.
            filename = Path(str(data.get("filename", "output.txt"))).name or "output.txt"
            if Path(filename).suffix.lower() not in self.ALLOWED_SUFFIXES:
                filename += ".txt"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            target = self.output_dir / filename
            target.write_text(str(data.get("content", "")), encoding="utf-8")
            return ToolResult(True, f"Saved {target.stat().st_size} bytes to {target}")
        except (json.JSONDecodeError, OSError, AttributeError) as exc:
            return ToolResult(False, f"Error writing file: {exc}")


# ==========================================================================
# 2. REASONING - planners decide the next action from the goal + history
# ==========================================================================


@dataclass
class Step:
    index: int
    thought: str
    tool: str
    tool_input: str
    observation: str = ""
    ok: bool = True


@dataclass
class Action:
    thought: str
    tool: Optional[str] = None  # None => finish
    tool_input: str = ""
    final_answer: Optional[str] = None


class Planner(ABC):
    @abstractmethod
    def next_action(self, goal: str, steps: List[Step]) -> Action: ...


_EXPR_RE = re.compile(r"[\d(][\d\s.+\-*/%^()]*")
_MATH_WORDS = re.compile(r"\b(calculate|compute|evaluate|solve|what is|what's)\b", re.I)
_SEARCH_WORDS = re.compile(r"\b(search|find|look up|trends?|latest)\b", re.I)
_SAVE_WORDS = re.compile(r"\b(save|write|store|export)\b", re.I)


def extract_expression(text: str) -> Optional[str]:
    """Longest arithmetic-looking span in the text that contains an operator."""
    spans = [m.group().strip() for m in _EXPR_RE.finditer(text)]
    spans = [s for s in spans if re.search(r"[+\-*/%^]", s)]
    return max(spans, key=len) if spans else None


def extract_query(goal: str) -> str:
    q = re.split(r"\s*(?:,|\band\b|\bthen\b)\s*(?:save|write|store|export)\b", goal, flags=re.I)[0]
    q = re.sub(r"^.*?\b(search|find|look up)(\s+for)?\s+", "", q, flags=re.I)
    return q.strip(" .,") or goal


class RuleBasedPlanner(Planner):
    """Deterministic planner. Order of sub-tasks: calculate -> search -> save."""

    def _intents(self, goal: str) -> List[str]:
        intents = []
        expr = extract_expression(goal)
        if expr and (_MATH_WORDS.search(goal) or re.fullmatch(r"[\d\s.+\-*/%^()?=]+", goal.strip())):
            intents.append("calculator")
        if _SEARCH_WORDS.search(goal):
            intents.append("web_search")
        if _SAVE_WORDS.search(goal):
            intents.append("file_writer")
        return intents

    def next_action(self, goal: str, steps: List[Step]) -> Action:
        attempted = {s.tool for s in steps}
        pending = [t for t in self._intents(goal) if t not in attempted]

        if not pending:
            if not steps:
                return Action("No tool is needed; I can answer from internal reasoning.", final_answer=(
                    "I did not find a task that needs a tool (calculation, search or saving a file)."))
            return Action("All required sub-tasks are done, so I can answer.")

        tool = pending[0]
        if tool == "calculator":
            return Action("The goal contains arithmetic, so I will use the calculator.",
                          tool, extract_expression(goal) or "")
        if tool == "web_search":
            return Action("The goal needs external knowledge, so I will search.", tool, extract_query(goal))
        # file_writer: save everything observed so far
        name = re.search(r"([\w\-]+\.(?:txt|md|json|csv))", goal)
        lines = [f"Goal: {goal}", ""]
        lines += [f"[{s.tool}] {s.observation}" for s in steps] or ["(no earlier observations)"]
        payload = json.dumps({"filename": name.group(1) if name else "agent_report.txt",
                              "content": "\n".join(lines)})
        return Action("The goal asks to persist results, so I will write them to a file.", tool, payload)


class LLMPlanner(Planner):
    """Optional: lets a Claude model choose the next action. Falls back to rules on any failure."""

    def __init__(self, tools: Dict[str, Tool], model: str = "claude-sonnet-5-5",
                 fallback: Optional[Planner] = None):
        self.tools, self.model = tools, model
        self.fallback = fallback or RuleBasedPlanner()

    def next_action(self, goal: str, steps: List[Step]) -> Action:
        try:
            import anthropic  # pip install anthropic

            tool_doc = "\n".join(f"- {t.name}: {t.description}" for t in self.tools.values())
            history = "\n".join(f"{s.index}. {s.tool}({s.tool_input}) -> {s.observation}" for s in steps) or "(none)"
            system = (
                "You are the reasoning module of a ReAct agent. Choose the single next action.\n"
                f"Tools:\n{tool_doc}\n"
                'Reply with JSON only: {"thought": str, "tool": str|null, "tool_input": str, "final_answer": str|null}. '
                "Set tool to null and give final_answer when the goal is complete. "
                "file_writer's tool_input must be a JSON string with 'filename' and 'content'."
            )
            resp = anthropic.Anthropic().messages.create(
                model=self.model, max_tokens=400, system=system,
                messages=[{"role": "user", "content": f"Goal: {goal}\nHistory so far:\n{history}"}],
            )
            raw = resp.content[0].text
            data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            tool = data.get("tool")
            if tool is not None and tool not in self.tools:
                raise ValueError(f"unknown tool {tool!r}")
            return Action(data.get("thought", ""), tool, str(data.get("tool_input", "")), data.get("final_answer"))
        except Exception as exc:  # noqa: BLE001 - any failure -> deterministic fallback
            action = self.fallback.next_action(goal, steps)
            action.thought = f"(LLM unavailable: {type(exc).__name__}; using rules) {action.thought}"
            return action


# ==========================================================================
# 3. ORCHESTRATOR - the ReAct loop
# ==========================================================================


class AutonomousAgent:
    def __init__(self, planner: Optional[Planner] = None, tools: Optional[List[Tool]] = None,
                 max_steps: int = 6, verbose: bool = True, output_dir: str = "agent_output"):
        tool_list = tools or [CalculatorTool(), MockSearchTool(), FileWriterTool(output_dir)]
        self.tools: Dict[str, Tool] = {t.name: t for t in tool_list}
        self.planner = planner or RuleBasedPlanner()
        self.max_steps = max_steps
        self.verbose = verbose
        self.memory: List[Dict[str, str]] = []  # conversation-style log across goals

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def run(self, goal: str) -> str:
        self._log(f"\n[Agent] Goal: {goal}\n" + "=" * 60)
        self.memory.append({"role": "user", "content": goal})
        steps: List[Step] = []

        for i in range(1, self.max_steps + 1):
            action = self.planner.next_action(goal, steps)
            self._log(f"[Thought {i}] {action.thought}")

            if action.tool is None:  # planner says we're done
                answer = action.final_answer or self._compose_answer(steps)
                break

            tool = self.tools.get(action.tool)
            if tool is None:
                result = ToolResult(False, f"Unknown tool '{action.tool}'")
            else:
                self._log(f"[Action {i}] {action.tool}({action.tool_input[:80]})")
                result = tool.run(action.tool_input)

            step = Step(i, action.thought, action.tool, action.tool_input, result.output, result.ok)
            steps.append(step)
            self.memory.append({"role": "tool", "content": f"{step.tool}: {step.observation}"})
            self._log(f"[Observation {i}] {'OK ' if result.ok else 'FAILED '}{result.output}")
        else:
            answer = "Stopped: step limit reached.\n" + self._compose_answer(steps)

        self.memory.append({"role": "assistant", "content": answer})
        self._log(f"[Final Answer]\n{answer}\n" + "=" * 60)
        return answer

    execute_task = run  # backwards-compatible name

    @staticmethod
    def _compose_answer(steps: List[Step]) -> str:
        if not steps:
            return "Agent Execution Complete. No tools were needed."
        lines = ["Agent Execution Complete."]
        lines += [f"  - {s.tool}: {s.observation}" for s in steps]
        failed = [s.tool for s in steps if not s.ok]
        if failed:
            lines.append(f"  (Some steps failed: {', '.join(failed)})")
        return "\n".join(lines)


# ==========================================================================
# 4. ENTRY POINT
# ==========================================================================

DEMO_GOALS = [
    "calculate 500 + 500",
    "search AI agent trends 2026",
    "save research summary to file",
    "calculate (12 + 8) * 5, search AI agent trends 2026 and save the results to report.txt",
]


def build_agent(**kwargs) -> AutonomousAgent:
    agent = AutonomousAgent(**kwargs)
    if os.getenv("AGENT_PLANNER", "rules").lower() == "llm":
        agent.planner = LLMPlanner(agent.tools)
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous ReAct agent")
    parser.add_argument("goal", nargs="*", help="goal to run (omit to run the demo)")
    parser.add_argument("-i", "--interactive", action="store_true", help="prompt for goals until 'quit'")
    args = parser.parse_args()
    agent = build_agent()

    if args.interactive:
        while True:
            goal = input("Enter goal (or 'quit'): ").strip()
            if goal.lower() in {"quit", "exit", "q"}:
                break
            if goal:
                agent.run(goal)
    elif args.goal:
        agent.run(" ".join(args.goal))
    else:
        for goal in DEMO_GOALS:
            agent.run(goal)


if __name__ == "__main__":
    main()
