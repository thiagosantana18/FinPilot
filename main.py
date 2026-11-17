# Projeto desenvolvido com o auxílio da IA Claude Sonnet 5.5 (Anthropic).
"""Interface de linha de comando do FinPilot."""
import sys

from finpilot import config
from finpilot.agent import ReActAgent, Step
from finpilot.storage import TransactionStore
from finpilot.tools import build_tools

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
HELP = """Comandos: /ajuda  /raciocinio (liga/desliga o raciocínio ReAct)  /limpar (zera a memória)  /sair
Exemplos:
  Gastei R$ 85 em mercado e R$ 40 em transporte
  Quanto gastei com alimentação este mês?
  Quero juntar R$ 12.000 em 18 meses, quanto devo guardar por mês?
  E se eu aplicar isso a 1% ao mês?"""


def print_step(step: Step) -> None:
    """Mostra um ciclo Thought/Action/Observation em cinza."""
    print(f"{DIM}  Thought: {step.thought}\n  Action: {step.action} {step.action_input}\n"
          f"  Observation: {step.observation}{RESET}")


def create_llm():
    """Cria o LLM local via Ollama (import tardio para dar erro amigável se faltar o pacote)."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        sys.exit("Pacote 'langchain-ollama' não encontrado. Rode: pip install -r requirements.txt")
    return ChatOllama(model=config.MODEL_NAME, base_url=config.OLLAMA_URL, temperature=config.TEMPERATURE)


def main() -> None:
    show_reasoning = True
    agent = ReActAgent(
        llm=create_llm(),
        tools=build_tools(TransactionStore(config.DATA_FILE)),
        max_steps=config.MAX_STEPS,
        history_window=config.HISTORY_WINDOW,
        on_step=lambda step: print_step(step) if show_reasoning else None,
    )

    print(f"{BOLD}FinPilot{RESET} (modelo: {config.MODEL_NAME}). Digite /ajuda para ver os comandos.\n")
    while True:
        try:
            user_input = input(f"{BOLD}Você> {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAté logo!")
            break
        if not user_input:
            continue
        if user_input == "/sair":
            print("Até logo!")
            break
        if user_input == "/ajuda":
            print(HELP)
        elif user_input == "/limpar":
            agent.reset_memory()
            print("Memória da conversa limpa.")
        elif user_input == "/raciocinio":
            show_reasoning = not show_reasoning
            print(f"Exibição do raciocínio: {'ligada' if show_reasoning else 'desligada'}.")
        else:
            try:
                print(f"{BOLD}FinPilot>{RESET} {agent.run(user_input).answer}\n")
            except Exception as exc:  # noqa: BLE001 - a CLI não deve cair por falha do LLM/rede
                print(f"Erro ao falar com o modelo ({exc}).\n"
                      f"Verifique se o Ollama está ativo ('ollama serve') e se o modelo "
                      f"'{config.MODEL_NAME}' foi baixado ('ollama pull {config.MODEL_NAME}').\n")


if __name__ == "__main__":
    main()
