
import sys
import pytest
from unittest import mock

@pytest.mark.integration
def test_cli_full_interaction(monkeypatch):
    """
    Teste de integração completo do CLI simulando chat, memória, debug e comandos especiais.
    Utiliza mocks para evitar chamadas reais a LLM/RAG.
    """
    # Sequência de comandos simulando um usuário real
    user_inputs = [
        'Qual o tipo do Pikachu?\n',
        'memória\n',
        'debug\n',
        'rag\n',
        'limpar_memória\n',
        'memória\n',
        'debug\n',
        'sair\n',
    ]
    # Função auxiliar para simular input do usuário
    def input_side_effect(*args, **kwargs):
        if user_inputs:
            return user_inputs.pop(0)
        return 'sair\n'

    # Mock das respostas do agente e do RAG
    class DummyAgent:
        def __init__(self):
            self.memory = DummyMemory()
        def respond(self, prompt):
            class DummyResp:
                content = 'Pikachu é do tipo Elétrico.'
            return DummyResp()
    class DummyMemory:
        def __init__(self):
            self._history = []
        def get_history(self):
            return [
                {'role': 'user', 'content': 'Qual o tipo do Pikachu?'},
                {'role': 'assistant', 'content': 'Pikachu é do tipo Elétrico.'}
            ] if self._history or 'memória' in user_inputs else []
        def clear(self):
            self._history = []
    class DummyRAG:
        def retrieve(self, query):
            return []
        def format_results(self, results):
            return ''
        def count_tokens(self, text):
            return 10

    # Patch instâncias do CLI para usar mocks
    import pokeconsultor.ui.cli as cli_mod
    monkeypatch.setattr(cli_mod, 'PokeConsultorCLI', cli_mod.PokeConsultorCLI)
    monkeypatch.setattr(cli_mod, 'HumanMessage', lambda content: content)
    
    # Instancia CLI com mocks
    cli = cli_mod.PokeConsultorCLI(DummyAgent(), DummyRAG())
    cli.print_header = lambda: None  # Evita prints de header
    
    # Patch sys.stdin para simular entrada do usuário
    monkeypatch.setattr(sys, 'stdin', mock.Mock())
    sys.stdin.readline = lambda: input_side_effect()

    # Executa o loop principal (run) até o comando 'sair'
    cli.run()
    # Se chegou até aqui, todos comandos foram processados sem erro
    assert True
