"""Agente ReAct: alterna entre raciocinar (Thought), agir (Action) e observar (Observation)."""
import json
from dataclasses import dataclass, field
from typing import Callable

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool

from .parser import parse_llm_output
from .prompts import build_system_prompt


@dataclass
class Step:
    """Um ciclo do raciocínio ReAct."""
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResponse:
    answer: str
    steps: list[Step] = field(default_factory=list)


class ReActAgent:
    """Agente com ferramentas LangChain, laço ReAct explícito e memória conversacional."""

    def __init__(self, llm, tools: list[BaseTool], max_steps: int = 6, history_window: int = 10,
                 on_step: Callable[[Step], None] | None = None):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.history_window = history_window
        self.on_step = on_step  # callback usado pela CLI para mostrar o raciocínio
        self.system_prompt = build_system_prompt(tools)
        self.memory = InMemoryChatMessageHistory()  # guarda só perguntas e respostas finais

    def reset_memory(self) -> None:
        self.memory.clear()

    def run(self, user_input: str) -> AgentResponse:
        """Responde a uma pergunta executando o ciclo ReAct até a Final Answer."""
        scratchpad = ""  # Thought/Action/Observation desta pergunta
        steps: list[Step] = []

        for _ in range(self.max_steps):
            messages = [
                SystemMessage(content=self.system_prompt),
                *self.memory.messages,
                HumanMessage(content=f"Question: {user_input}\n{scratchpad}"),
            ]
            # O stop evita que o modelo "invente" a Observation por conta própria.
            raw = self.llm.invoke(messages, stop=["\nObservation:", "Observation:"]).content
            parsed = parse_llm_output(raw)

            if parsed.kind == "final":
                return self._finish(user_input, parsed.answer, steps)

            if parsed.kind == "invalid":
                observation = ("Formato inválido. Responda com 'Thought:' seguido de 'Action:' e "
                               "'Action Input:' (JSON), ou com 'Final Answer:'.")
                step = Step(parsed.thought or raw.strip(), "-", "-", observation)
            else:
                observation = self._execute_tool(parsed.action, parsed.action_input)
                step = Step(parsed.thought, parsed.action, parsed.action_input, observation)

            steps.append(step)
            if self.on_step:
                self.on_step(step)
            scratchpad += (f"Thought: {step.thought}\nAction: {step.action}\n"
                           f"Action Input: {step.action_input}\nObservation: {step.observation}\n")

        answer = "Não consegui concluir a tarefa dentro do limite de passos. Pode reformular o pedido?"
        return self._finish(user_input, answer, steps)

    def _execute_tool(self, name: str, raw_input: str) -> str:
        """Executa a ferramenta; erros viram Observation para o agente poder se corrigir."""
        tool = self.tools.get(name)
        if tool is None:
            return f"Erro: a ferramenta '{name}' não existe. Use uma de: {', '.join(self.tools)}."
        try:
            args = json.loads(raw_input)
            if not isinstance(args, dict):
                raise ValueError("Action Input deve ser um objeto JSON.")
            return str(tool.invoke(args))
        except Exception as exc:  # noqa: BLE001 - qualquer falha deve voltar ao agente como texto
            return f"Erro ao executar '{name}': {exc}"

    def _finish(self, question: str, answer: str, steps: list[Step]) -> AgentResponse:
        """Grava o par pergunta/resposta na memória (com janela deslizante) e devolve o resultado."""
        self.memory.add_user_message(question)
        self.memory.add_ai_message(answer)
        self.memory.messages = self.memory.messages[-self.history_window:]
        return AgentResponse(answer=answer, steps=steps)
