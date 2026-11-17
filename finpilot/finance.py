"""Cálculos financeiros puros (sem dependência de LLM ou LangChain), fáceis de testar."""
from collections import defaultdict


def monthly_contribution(goal: float, months: int, monthly_rate: float = 0.0) -> float:
    """Aporte mensal (no fim de cada mês) para atingir `goal` em `months` meses.

    `monthly_rate` é decimal (0.01 = 1% ao mês).
    """
    if goal <= 0:
        raise ValueError("O valor da meta deve ser maior que zero.")
    if months <= 0:
        raise ValueError("O prazo deve ser de pelo menos 1 mês.")
    if monthly_rate < 0:
        raise ValueError("A taxa mensal não pode ser negativa.")
    if monthly_rate == 0:
        return goal / months
    return goal * monthly_rate / ((1 + monthly_rate) ** months - 1)


def compound_growth(initial: float, contribution: float, monthly_rate: float, months: int) -> float:
    """Saldo final após `months` meses com aporte mensal e juros compostos."""
    if months <= 0:
        raise ValueError("O prazo deve ser de pelo menos 1 mês.")
    if initial < 0 or contribution < 0 or monthly_rate < 0:
        raise ValueError("Valores e taxa não podem ser negativos.")
    balance = initial
    for _ in range(months):
        balance = balance * (1 + monthly_rate) + contribution
    return balance


def totals_by_category(transactions: list[dict]) -> dict[str, float]:
    """Soma os valores das transações agrupando por categoria (ordem decrescente)."""
    totals: dict[str, float] = defaultdict(float)
    for tx in transactions:
        totals[tx["category"]] += tx["amount"]
    return dict(sorted(totals.items(), key=lambda item: item[1], reverse=True))


def format_brl(value: float) -> str:
    """Formata como moeda brasileira: 1234.5 -> 'R$ 1.234,50'."""
    text = f"{value:,.2f}"
    return "R$ " + text.replace(",", "X").replace(".", ",").replace("X", ".")
