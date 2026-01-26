from unittest import mock

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot

from pokeconsultor.ui.gui import PokeConsultorGUI


@pytest.mark.integration
@pytest.mark.usefixtures("qtbot")
def test_gui_basic_interaction(qtbot: QtBot):
    """
    Teste básico da GUI: simula envio de pergunta e verifica resposta mockada.
    """

    # Mock do agente e rag_service
    class DummyAgent:
        def __init__(self):
            self.memory = DummyMemory()

        def respond(self, prompt):
            class DummyResp:
                content = "Pikachu é do tipo Elétrico."

            return DummyResp()

    class DummyMemory:
        def clear(self):
            pass

    class DummyRAG:
        def retrieve(self, query):
            return []

        def format_results(self, results):
            return ""

        def count_tokens(self, text):
            return 10

    # Instancia a janela
    gui = PokeConsultorGUI(DummyAgent(), DummyRAG())
    qtbot.addWidget(gui)
    gui.show()

    # Simula digitação e envio de pergunta

    qtbot.keyClicks(gui.input_field, "Qual o tipo do Pikachu?")
    qtbot.keyPress(gui.input_field, Qt.Key_Return)

    # Aguarda processamento da thread
    qtbot.waitUntil(
        lambda: "Pikachu é do tipo Elétrico." in gui.chat_display.toPlainText(),
        timeout=3000,
    )

    # Verifica se a resposta mockada apareceu
    assert "Pikachu é do tipo Elétrico." in gui.chat_display.toPlainText()
