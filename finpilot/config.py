"""Configurações centralizadas (podem ser sobrescritas por variáveis de ambiente)."""
import os
from pathlib import Path

# Modelo local servido pelo Ollama. Leve por padrão; "qwen2.5:7b" segue melhor o formato ReAct.
MODEL_NAME = os.getenv("FINPILOT_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TEMPERATURE = 0.0  # determinístico: ajuda modelos pequenos a respeitar o formato

# Arquivo local onde as transações são persistidas.
DATA_FILE = Path(os.getenv("FINPILOT_DATA", Path(__file__).parent.parent / "data" / "transacoes.json"))

MAX_STEPS = 6        # limite de ciclos Thought/Action/Observation por pergunta
HISTORY_WINDOW = 10  # quantidade de mensagens da conversa mantidas na memória
