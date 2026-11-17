"""Ferramentas (tools) LangChain do FinPilot.

Cada tool tem nome, descrição e schema de argumentos validado por Pydantic.
Toda a matemática e persistência acontece aqui, em Python, e não no LLM.
"""
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .finance import compound_growth, format_brl, monthly_contribution, totals_by_category
from .storage import TransactionStore, resolve_month


class RegistrarInput(BaseModel):
    tipo: str = Field(description="'despesa' ou 'receita'")
    valor: float = Field(description="Valor em reais, ex.: 85.5")
    categoria: str = Field(description="Categoria, ex.: alimentacao, transporte, salario")
    descricao: str = Field(default="", description="Descrição curta (opcional)")
    data: str | None = Field(default=None, description="YYYY-MM-DD (opcional; padrão: hoje)")


class ConsultarInput(BaseModel):
    tipo: str = Field(default="despesa", description="'despesa' ou 'receita'")
    categoria: str | None = Field(default=None, description="Filtra por categoria (opcional)")
    mes: str | None = Field(default="atual", description="'YYYY-MM', 'atual' ou 'todos'")


class MetaInput(BaseModel):
    valor_meta: float = Field(description="Quanto se quer juntar, em reais")
    meses: int = Field(description="Prazo em meses")
    taxa_mensal_percentual: float = Field(default=0.0, description="Rendimento mensal em %, ex.: 1 para 1%")


class JurosInput(BaseModel):
    valor_inicial: float = Field(default=0.0, description="Valor já investido, em reais")
    aporte_mensal: float = Field(default=0.0, description="Aporte mensal, em reais")
    taxa_mensal_percentual: float = Field(description="Rendimento mensal em %, ex.: 1 para 1%")
    meses: int = Field(description="Prazo em meses")


def build_tools(store: TransactionStore) -> list[StructuredTool]:
    """Cria as quatro ferramentas ligadas ao `store` informado."""

    def registrar_transacao(tipo: str, valor: float, categoria: str, descricao: str = "",
                            data: str | None = None) -> str:
        tx = store.add(tipo, valor, categoria, descricao, data)
        return f"Registrado: {tx['kind']} de {format_brl(tx['amount'])} em '{tx['category']}' ({tx['date']})."

    def consultar_gastos(tipo: str = "despesa", categoria: str | None = None, mes: str | None = "atual") -> str:
        period = resolve_month(mes)
        found = store.query(kind=tipo, category=categoria, month=period)
        if not found:
            return "Nenhuma transação encontrada para esse filtro."
        total = sum(t["amount"] for t in found)
        breakdown = "; ".join(f"{cat}: {format_brl(v)}" for cat, v in totals_by_category(found).items())
        return (f"Total de {tipo}s ({period or 'todos os meses'}): {format_brl(total)} "
                f"em {len(found)} lançamento(s). Por categoria -> {breakdown}.")

    def simular_meta(valor_meta: float, meses: int, taxa_mensal_percentual: float = 0.0) -> str:
        aporte = monthly_contribution(valor_meta, meses, taxa_mensal_percentual / 100)
        return (f"Para juntar {format_brl(valor_meta)} em {meses} meses com rendimento de "
                f"{taxa_mensal_percentual:g}% ao mês, o aporte mensal é {format_brl(aporte)}.")

    def juros_compostos(taxa_mensal_percentual: float, meses: int, valor_inicial: float = 0.0,
                        aporte_mensal: float = 0.0) -> str:
        final = compound_growth(valor_inicial, aporte_mensal, taxa_mensal_percentual / 100, meses)
        invested = valor_inicial + aporte_mensal * meses
        return (f"Após {meses} meses a {taxa_mensal_percentual:g}% ao mês: saldo {format_brl(final)} "
                f"(investido {format_brl(invested)}, juros {format_brl(final - invested)}).")

    return [
        StructuredTool.from_function(
            func=registrar_transacao, name="registrar_transacao", args_schema=RegistrarInput,
            description=("Salva uma despesa ou receita. Campos: tipo, valor, categoria, descricao (opcional), "
                         "data (opcional, YYYY-MM-DD).")),
        StructuredTool.from_function(
            func=consultar_gastos, name="consultar_gastos", args_schema=ConsultarInput,
            description=("Soma despesas ou receitas já registradas. Campos: tipo, categoria (opcional), "
                         "mes ('YYYY-MM', 'atual' ou 'todos').")),
        StructuredTool.from_function(
            func=simular_meta, name="simular_meta", args_schema=MetaInput,
            description=("Calcula o aporte mensal para atingir uma meta. Campos: valor_meta, meses, "
                         "taxa_mensal_percentual (opcional, 0 se não houver rendimento).")),
        StructuredTool.from_function(
            func=juros_compostos, name="juros_compostos", args_schema=JurosInput,
            description=("Projeta o saldo de um investimento com juros compostos. Campos: taxa_mensal_percentual, "
                         "meses, valor_inicial (opcional), aporte_mensal (opcional).")),
    ]
