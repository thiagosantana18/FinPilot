"""Interpretação da saída textual do LLM no formato ReAct (sem dependência de LangChain)."""
import json
import re
from dataclasses import dataclass

_ACTION_RE = re.compile(r"Action:\s*(?P<name>[\w\-]+)\s*\n+\s*Action Input:\s*(?P<input>.*)", re.S)
_FINAL_RE = re.compile(r"Final Answer:\s*(?P<answer>.*)", re.S)
_THOUGHT_RE = re.compile(r"Thought:\s*(.*?)(?=\n\s*(?:Action:|Final Answer:)|\Z)", re.S)


@dataclass
class ParsedOutput:
    kind: str  # "action", "final" ou "invalid"
    thought: str = ""
    action: str = ""
    action_input: str = ""
    answer: str = ""


def _first_json_object(text: str) -> str:
    """Extrai o primeiro objeto JSON do texto; se falhar, devolve o texto limpo."""
    start = text.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            pass
    return text.strip()


def parse_llm_output(text: str) -> ParsedOutput:
    """Classifica a saída do modelo como ação, resposta final ou formato inválido."""
    # Modelos pequenos às vezes inventam a Observation: descartamos tudo depois dela.
    text = text.split("Observation:")[0]

    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group(1).strip() if thought_match else ""
    action = _ACTION_RE.search(text)
    final = _FINAL_RE.search(text)

    if action and (not final or action.start() < final.start()):
        return ParsedOutput("action", thought, action.group("name"), _first_json_object(action.group("input")))
    if final:
        return ParsedOutput("final", thought, answer=final.group("answer").strip())
    return ParsedOutput("invalid", thought)
