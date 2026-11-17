"""Teste do laço ReAct e da memória com um LLM roteirizado (sem Ollama)."""
import pytest

pytest.importorskip("langchain_core")

from finpilot.agent import ReActAgent  # noqa: E402
from finpilot.storage import TransactionStore  # noqa: E402
from finpilot.tools import build_tools  # noqa: E402


class ScriptedLLM:
    """Devolve respostas pré-definidas e guarda as mensagens recebidas."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages, stop=None):
        self.calls.append(messages)
        return type("Reply", (), {"content": self.replies.pop(0)})()


def test_ciclo_react_com_ferramenta_e_memoria(tmp_path):
    llm = ScriptedLLM([
        'Thought: calcular\nAction: simular_meta\nAction Input: {"valor_meta": 12000, "meses": 18}',
        "Thought: pronto\nFinal Answer: Guarde R$ 666,67 por mês.",
        'Thought: usar valores lembrados\nAction: juros_compostos\n'
        'Action Input: {"taxa_mensal_percentual": 1, "meses": 18, "aporte_mensal": 666.67}',
        "Thought: ok\nFinal Answer: Com 1% você teria mais.",
    ])
    agent = ReActAgent(llm, build_tools(TransactionStore(tmp_path / "t.json")))

    first = agent.run("Quero juntar 12000 em 18 meses")
    assert "666,67" in first.steps[0].observation and "666,67" in first.answer

    agent.run("E se eu aplicar a 1% ao mês?")
    history_text = " ".join(str(m.content) for m in llm.calls[2])
    assert "Quero juntar 12000" in history_text  # a memória chegou ao prompt


def test_erro_de_ferramenta_volta_como_observation(tmp_path):
    llm = ScriptedLLM([
        'Thought: x\nAction: ferramenta_fantasma\nAction Input: {}',
        'Thought: y\nAction: simular_meta\nAction Input: {"valor_meta": 100, "meses": 0}',
        "Final Answer: Prazo inválido.",
    ])
    agent = ReActAgent(llm, build_tools(TransactionStore(tmp_path / "t.json")))
    result = agent.run("teste")
    assert "não existe" in result.steps[0].observation
    assert "Erro" in result.steps[1].observation
