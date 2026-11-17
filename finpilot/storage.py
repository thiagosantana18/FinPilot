"""Persistência das transações em um arquivo JSON local."""
import json
import re
from datetime import date
from pathlib import Path

VALID_KINDS = ("despesa", "receita")


class StorageError(Exception):
    """Erro ao ler ou gravar o arquivo de transações."""


def resolve_month(value: str | None) -> str | None:
    """Converte 'atual' / 'todos' / 'YYYY-MM' em um filtro de mês (ou None para todos)."""
    if value is None or value.strip().lower() in ("", "todos", "tudo"):
        return None
    if value.strip().lower() == "atual":
        return date.today().strftime("%Y-%m")
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value.strip()):
        raise ValueError("Mês inválido. Use 'YYYY-MM', 'atual' ou 'todos'.")
    return value.strip()


class TransactionStore:
    """Guarda e consulta receitas e despesas."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise StorageError(f"Não foi possível ler {self.path}: {exc}") from exc

    def _save(self, transactions: list[dict]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(transactions, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            raise StorageError(f"Não foi possível gravar {self.path}: {exc}") from exc

    def add(self, kind: str, amount: float, category: str, description: str = "", when: str | None = None) -> dict:
        """Registra uma transação e a devolve."""
        kind = kind.strip().lower()
        if kind not in VALID_KINDS:
            raise ValueError("O tipo deve ser 'despesa' ou 'receita'.")
        if amount <= 0:
            raise ValueError("O valor deve ser maior que zero.")
        if not category.strip():
            raise ValueError("Informe uma categoria.")
        when = when or date.today().isoformat()
        try:
            date.fromisoformat(when)
        except ValueError as exc:
            raise ValueError("Data inválida. Use o formato YYYY-MM-DD.") from exc

        tx = {"kind": kind, "amount": round(float(amount), 2), "category": category.strip().lower(),
              "description": description.strip(), "date": when}
        transactions = self._load()
        transactions.append(tx)
        self._save(transactions)
        return tx

    def query(self, kind: str | None = None, category: str | None = None, month: str | None = None) -> list[dict]:
        """Filtra transações por tipo, categoria e/ou mês ('YYYY-MM')."""
        result = self._load()
        if kind:
            result = [t for t in result if t["kind"] == kind.strip().lower()]
        if category:
            result = [t for t in result if t["category"] == category.strip().lower()]
        if month:
            result = [t for t in result if t["date"].startswith(month)]
        return result
