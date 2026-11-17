"""Testes unitários: cálculos, armazenamento e parser (não exigem LLM)."""
import pytest

from finpilot.finance import compound_growth, format_brl, monthly_contribution
from finpilot.parser import parse_llm_output
from finpilot.storage import TransactionStore, resolve_month


def test_meta_sem_juros():
    assert monthly_contribution(12000, 18) == pytest.approx(666.67, abs=0.01)


def test_meta_com_juros():
    assert monthly_contribution(12000, 18, 0.01) == pytest.approx(611.78, abs=0.01)


def test_meta_e_juros_sao_consistentes():
    aporte = monthly_contribution(12000, 18, 0.01)
    assert compound_growth(0, aporte, 0.01, 18) == pytest.approx(12000, abs=0.01)


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        monthly_contribution(1000, 0)
    with pytest.raises(ValueError):
        compound_growth(0, 100, -0.01, 12)


def test_formato_brl():
    assert format_brl(1234.5) == "R$ 1.234,50"


def test_armazenamento_e_filtros(tmp_path):
    store = TransactionStore(tmp_path / "t.json")
    store.add("despesa", 85, "Alimentacao", when="2026-03-05")
    store.add("despesa", 40, "transporte", when="2026-03-06")
    store.add("despesa", 30, "alimentacao", when="2026-04-01")
    assert sum(t["amount"] for t in store.query(category="alimentacao", month="2026-03")) == 85
    assert len(store.query(kind="despesa")) == 3


def test_armazenamento_valida_dados(tmp_path):
    store = TransactionStore(tmp_path / "t.json")
    with pytest.raises(ValueError):
        store.add("gasto", 10, "x")
    with pytest.raises(ValueError):
        store.add("despesa", 10, "x", when="05/03/2026")


def test_resolve_month():
    assert resolve_month("todos") is None
    assert resolve_month("2026-03") == "2026-03"
    with pytest.raises(ValueError):
        resolve_month("março")


def test_parser_acao():
    out = parse_llm_output('Thought: registrar\nAction: simular_meta\nAction Input: {"valor_meta": 1, "meses": 2}\nObservation: inventada')
    assert out.kind == "action" and out.action == "simular_meta"
    assert '"meses": 2' in out.action_input


def test_parser_final_e_invalido():
    assert parse_llm_output("Thought: ok\nFinal Answer: Pronto!").answer == "Pronto!"
    assert parse_llm_output("blablabla").kind == "invalid"
