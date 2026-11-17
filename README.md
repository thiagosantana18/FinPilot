# FinPilot: Assistente de Finanças Pessoais com Agente ReAct

Agente de linha de comando que conversa em português, registra gastos, consulta totais e simula metas e
investimentos. Usa **LangChain**, um **LLM local e leve (Ollama)**, o padrão **ReAct** e **memória conversacional**.

## 1. Problema e solução

**Problema:** pessoas que querem organizar o dinheiro esbarram em planilhas e calculadoras separadas.
LLMs sozinhos erram contas e não guardam dados; já uma planilha não entende "gastei 85 no mercado".

**Solução:** um agente que entende linguagem natural, **decide qual ferramenta usar** e deixa a matemática e a
persistência para código Python determinístico.

- **Público-alvo:** jovens adultos e profissionais iniciantes em organização financeira.
- **Objetivo:** registrar, consultar e planejar finanças conversando.

## 2. Arquitetura

```
main.py (CLI) ──► ReActAgent ──► LLM local (ChatOllama)
                      │  ▲
                      │  └── Observation (resultado da tool)
                      ▼
               Tools (LangChain StructuredTool) ──► finance.py (cálculos) / storage.py (JSON)
                      
Memória: InMemoryChatMessageHistory (janela das últimas 10 mensagens)
```

| Arquivo | Responsabilidade |
|---|---|
| `main.py` | Interface CLI e comandos |
| `finpilot/agent.py` | Laço ReAct, execução das tools e memória |
| `finpilot/parser.py` | Interpreta `Thought/Action/Action Input/Final Answer` |
| `finpilot/prompts.py` | Prompt de sistema com formato ReAct e exemplo (few-shot) |
| `finpilot/tools.py` | As 4 ferramentas LangChain (nome, descrição e schema Pydantic) |
| `finpilot/finance.py` | Cálculos puros (aporte, juros compostos, totais) |
| `finpilot/storage.py` | Persistência em `data/transacoes.json` |
| `finpilot/config.py` | Modelo, URL do Ollama, limites |
| `tests/` | Testes unitários e do laço ReAct |

### Ferramentas

| Tool | Descrição | Campos |
|---|---|---|
| `registrar_transacao` | Salva despesa/receita | tipo, valor, categoria, descricao?, data? |
| `consultar_gastos` | Soma lançamentos por categoria/mês | tipo, categoria?, mes (`YYYY-MM`/`atual`/`todos`) |
| `simular_meta` | Aporte mensal para atingir uma meta | valor_meta, meses, taxa_mensal_percentual? |
| `juros_compostos` | Projeta saldo de um investimento | taxa_mensal_percentual, meses, valor_inicial?, aporte_mensal? |

### Como o ReAct foi implementado

O laço em `ReActAgent.run` é explícito (sem caixa-preta), o que o torna fácil de estudar e funciona mesmo com
modelos pequenos que **não têm suporte nativo a function calling**:

1. O modelo recebe o prompt e escreve `Thought` + `Action` + `Action Input` (JSON).
2. O parâmetro `stop=["Observation:"]` impede que ele invente o resultado.
3. O agente executa a tool e acrescenta a `Observation` ao raciocínio.
4. O ciclo repete até surgir `Final Answer` ou até `MAX_STEPS`.
5. Erros (tool inexistente, JSON inválido, valor negativo) voltam como `Observation`, e o modelo pode se corrigir.

### Memória

Cada pergunta e resposta final são guardadas em `InMemoryChatMessageHistory` e reenviadas em toda chamada.
Só o par pergunta/resposta entra na memória (não o raciocínio intermediário), mantendo o contexto curto para
modelos pequenos. A janela é deslizante (10 mensagens). O comando `/limpar` zera a memória.

## 3. Como executar

**Pré-requisitos:** Python 3.10+ e [Ollama](https://ollama.com) instalado.

```bash
ollama pull qwen2.5:3b          # modelo leve (~2 GB). Alternativa mais precisa: qwen2.5:7b
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Para trocar o modelo: `FINPILOT_MODEL=qwen2.5:7b python main.py` (Windows PowerShell: `$env:FINPILOT_MODEL="qwen2.5:7b"`).

Comandos: `/ajuda`, `/raciocinio` (mostra/oculta Thought/Action/Observation), `/limpar`, `/sair`.

Testes: `pytest -q` (os testes do agente usam um LLM roteirizado, portanto não precisam do Ollama).

## 4. Casos de uso e resultados

| # | Entrada | Comportamento esperado do agente | Verificado |
|---|---|---|---|
| 1 | "Gastei R$ 85 em mercado e R$ 40 em transporte" | 2 chamadas a `registrar_transacao` | Lógica da tool (testes do núcleo) |
| 2 | "Quanto gastei com alimentação este mês?" | `consultar_gastos` com `mes: "atual"` | Lógica da tool |
| 3 | "Quero juntar R$ 12.000 em 18 meses" | `simular_meta` → **R$ 666,67/mês** | Cálculo conferido |
| 4 | "E se eu aplicar isso a 1% ao mês?" | Memória recupera 12.000 e 18 meses → `simular_meta` com taxa 1 → **R$ 611,78/mês**; ou `juros_compostos` com aporte 666,67 → **R$ 13.076,56** | Cálculos conferidos |

Sessão (caso 3 e 4):

```
Você> Quero juntar R$ 12.000 em 18 meses, quanto devo guardar por mês?
  Thought: Preciso calcular o aporte mensal necessário.
  Action: simular_meta {"valor_meta": 12000, "meses": 18}
  Observation: Para juntar R$ 12.000,00 em 18 meses com rendimento de 0% ao mês, o aporte mensal é R$ 666,67.
FinPilot> Você precisa guardar R$ 666,67 por mês.

Você> E se eu aplicar isso a 1% ao mês?
  Thought: "Isso" se refere à meta de R$ 12.000 em 18 meses, citada antes. Vou recalcular com 1%.
  Action: simular_meta {"valor_meta": 12000, "meses": 18, "taxa_mensal_percentual": 1}
  Observation: ... o aporte mensal é R$ 611,78.
FinPilot> Com rendimento de 1% ao mês, basta guardar R$ 611,78 por mês.
```

## 5. Reflexão: agente × abordagem manual

**Sem o agente**, o usuário teria de: (1) abrir uma planilha, escolher colunas e digitar cada gasto; (2) montar
filtros ou tabelas dinâmicas para somar por categoria e mês; (3) lembrar a fórmula de parcela/juros compostos ou
procurar uma calculadora online; (4) repetir tudo a cada nova hipótese ("e se fosse 1%?").

**Com o agente:**
- A **entrada é linguagem natural**: uma frase vira vários registros estruturados.
- O **ReAct** decide sozinho a sequência (registrar duas vezes, depois consultar) e mostra o raciocínio, o que
  dá transparência e facilita depurar.
- As **ferramentas** garantem exatidão: o LLM não faz contas, só escolhe e interpreta.
- A **memória** elimina repetição: "e se eu aplicar isso a 1%?" funciona sem reinformar valor e prazo.

**Limitações:** modelos de 3B podem errar o formato ReAct ou a categoria (por isso há validação, mensagens de erro
devolvidas ao agente e o limite de passos); a memória é só da sessão (as transações, porém, persistem em disco);
o agente não substitui consultoria financeira.

## 6. Boas práticas adotadas

Módulos com responsabilidade única; cálculos puros separados da camada LangChain (testáveis sem LLM); validação
com Pydantic; erros tratados na CLI, no armazenamento e nas ferramentas; configuração por variáveis de ambiente.

---
*Este projeto foi desenvolvido com o auxílio da IA **Claude Sonnet 5.5** (Anthropic), utilizada como assistente
na implementação e documentação.*
