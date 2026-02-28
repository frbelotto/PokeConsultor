"""PySide6 GUI for PokeConsultor."""

import sys
import threading
from typing import Any

from langchain.messages import HumanMessage
from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pokeconsultor.services.logger import logger


class WorkerSignals(QObject):
    """Signals for the background worker."""

    response_ready = Signal(str, str)
    error_occurred = Signal(str)


class PokeConsultorGUI(QMainWindow):
    """PySide6 interface for PokeConsultor."""

    def __init__(self, agent: Any) -> None:
        super().__init__()
        self.agent = agent
        self.signals = WorkerSignals()
        self.signals.response_ready.connect(self._on_response_ready)
        self.signals.error_occurred.connect(self._on_error)
        self.counter: int = 0

        self.init_ui()

    def init_ui(self) -> None:
        """Initialize the GUI layout."""
        self.setWindowTitle("PokeConsultor - AI Assistant")
        self.resize(1000, 700)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Splitter for Chat and Debug
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left Side: Chat ---
        chat_container = QWidget()
        chat_layout = QVBoxLayout(chat_container)

        # Chat Display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Arial", 11))
        chat_layout.addWidget(self.chat_display)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Faça sua pergunta...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Enviar")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)

        # --- Right Side: Debug/Context (Side Panel) ---
        self.side_panel = QWidget()
        self.side_layout = QVBoxLayout(self.side_panel)

        self.side_label = QLabel("🔍 Informações do Agente / Debug")
        self.side_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.side_layout.addWidget(self.side_label)

        self.debug_display = QTextEdit()
        self.debug_display.setReadOnly(True)
        self.debug_display.setFont(QFont("Courier New", 10))
        self.side_layout.addWidget(self.debug_display)

        self.splitter.addWidget(chat_container)
        self.splitter.addWidget(self.side_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)

        main_layout.addWidget(self.splitter)

        # --- Bottom Toolbar: Toggles ---
        toolbar_layout = QHBoxLayout()

        self.rag_status_label = QLabel("RAG: automático (tool do agente)")
        self.rag_status_label.setStyleSheet(
            "padding: 4px 8px; border-radius: 6px; background-color: #2d3436; color: #dfe6e9;"
        )
        toolbar_layout.addWidget(self.rag_status_label)

        self.tool_usage_label = QLabel("Última interação: sem uso de tool")
        self.tool_usage_label.setStyleSheet("font-size: 12px; color: #636e72;")
        toolbar_layout.addWidget(self.tool_usage_label)

        self.debug_checkbox = QCheckBox("Modo Debug")
        self.debug_checkbox.setChecked(True)
        self.debug_checkbox.stateChanged.connect(self._toggle_debug_panel)
        toolbar_layout.addWidget(self.debug_checkbox)

        toolbar_layout.addStretch()

        button_style = """
            QPushButton {
                padding: 0 15px;
                font-weight: bold;
                font-size: 12px;
                height: 28px;
            }
            QPushButton::menu-indicator { image: none; }
        """

        # Help Button
        self.help_button = QPushButton("Ajuda")
        self.help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.help_button.setStyleSheet(button_style)

        help_menu = QMenu(self)
        help_menu.addAction("Manual (README)", self.show_help)
        help_menu.addAction("Código Fonte (GitHub)", self.open_repo)
        self.help_button.setMenu(help_menu)
        toolbar_layout.addWidget(self.help_button)

        # Clear Memory Button
        self.clear_mem_button = QPushButton("Limpar Memória")
        self.clear_mem_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_mem_button.setStyleSheet(button_style)
        self.clear_mem_button.clicked.connect(self.clear_memory)
        toolbar_layout.addWidget(self.clear_mem_button)

        main_layout.addLayout(toolbar_layout)

        # Initial message
        self._append_chat(
            (
                "🤖 <b>Sistema pronto!</b> Faça suas perguntas sobre Pokémon. "
                "A busca de contexto (RAG) é acionada automaticamente quando necessário."
            ),
            "system",
        )

    def show_help(self) -> None:
        """Show the help dialog with README content."""
        try:
            with open("README.md", "r", encoding="utf-8") as f:
                readme_content = f.read()
        except Exception:
            readme_content = "Não foi possível carregar o README.md"

        dialog = QDialog(self)
        dialog.setWindowTitle("Ajuda - PokeConsultor")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(readme_content)
        layout.addWidget(text_edit)

        close_button = QPushButton("Fechar")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)

        dialog.exec()

    def open_repo(self) -> None:
        """Open the repository URL in the default browser."""
        QDesktopServices.openUrl(QUrl("https://github.com/frbelotto/PokeConsultor"))

    def _toggle_debug_panel(self, state: int) -> None:
        """Show/hide the debug side panel."""
        self.side_panel.setVisible(state == Qt.CheckState.Checked.value)

    def _append_chat(self, text: str, sender: str) -> None:
        """Append text to the chat display."""
        # Use system text color as default
        default_color = self.palette().color(self.foregroundRole()).name()

        if sender == "user":
            color = "#3498db"  # A bright blue that works in both modes
            prefix = "👤 <b>Você:</b> "
        elif sender == "ai":
            color = default_color
            prefix = "✨ <b>IA:</b> "
        else:
            color = "#888888"
            prefix = ""

        self.chat_display.append(f"<div style='color: {color};'>{prefix}{text}</div>")
        self.chat_display.moveCursor(QTextCursor.MoveOperation.End)

    def send_message(self) -> None:
        """Handle sending a message."""
        query = self.input_field.text().strip()
        if not query:
            return

        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)

        self._append_chat(query, "user")

        # Run processing in a thread
        thread = threading.Thread(target=self._process_query, args=(query,))
        thread.daemon = True
        thread.start()

    def _process_query(self, query: str) -> None:
        """Background thread to process the query."""
        try:
            request = HumanMessage(content=query)
            logger.info(f"User prompt: {request}")
            response_text = self.agent.respond(prompt=request)

            # Garantir que a resposta seja string
            if not isinstance(response_text, str):
                try:
                    response_text = str(response_text.content)
                except Exception as e:
                    response_text = f"[ERRO ao converter resposta da IA: {e}]"

            debug_text = self._format_debug_info()

            self.signals.response_ready.emit(response_text, debug_text)

        except Exception as e:
            self.signals.error_occurred.emit(str(e))

    def _on_response_ready(self, response_text: str, debug_text: str) -> None:
        """Handle UI update when response is ready."""
        self._append_chat(response_text, "ai")

        # Update debug panel
        self.debug_display.setPlainText(debug_text)
        self._update_tool_usage_badge()

        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def _on_error(self, error_msg: str) -> None:
        """Handle errors during processing."""
        self._append_chat(f"❌ Erro: {error_msg}", "system")
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

    def _format_debug_info(self) -> str:
        """Format graph state history for the side panel."""
        history = self.agent.get_state_history()
        tool_usage = self.agent.get_latest_interaction_tool_usage()

        output = []
        output.append("=== ESTADOS RECENTES DO AGENTE ===")
        output.append(f"Total: {len(history)}\n")

        used = "SIM" if tool_usage["used"] else "NÃO"
        names = ", ".join(tool_usage["tool_names"]) if tool_usage["tool_names"] else "-"
        output.append("=== TOOL USAGE (ÚLTIMA INTERAÇÃO) ===")
        output.append(f"Tool usada: {used}")
        output.append(f"Tools: {names}")
        output.append(f"Quantidade de chamadas: {tool_usage['tool_calls']}\n")

        if not history:
            output.append("Nenhum estado disponível.")
            return "\n".join(output)

        for i, state in enumerate(history[:3], 1):
            output.append(f"[State {i}]")
            output.append(str(state))
            output.append("-" * 40)

        return "\n".join(output)

    def clear_memory(self) -> None:
        """Clear the AI agent memory and UI displays."""
        self.agent.clear_thread_memory()
        self.chat_display.clear()
        self.debug_display.clear()
        self.tool_usage_label.setText("Última interação: sem uso de tool")
        self._append_chat("🧠 Memória e histórico limpos!", "system")

    def _update_tool_usage_badge(self) -> None:
        """Update toolbar label with latest tool usage summary."""
        tool_usage = self.agent.get_latest_interaction_tool_usage()

        if tool_usage["used"]:
            names = (
                ", ".join(tool_usage["tool_names"])
                if tool_usage["tool_names"]
                else "tool"
            )
            self.tool_usage_label.setText(
                f"Última interação: usou {tool_usage['tool_calls']} chamada(s) ({names})"
            )
            self.tool_usage_label.setStyleSheet("font-size: 12px; color: #00b894;")
            return

        self.tool_usage_label.setText("Última interação: sem uso de tool")
        self.tool_usage_label.setStyleSheet("font-size: 12px; color: #636e72;")


def run_gui(agent: Any) -> None:
    """Entry point for the GUI version."""
    app = QApplication(sys.argv)
    window = PokeConsultorGUI(agent)
    window.show()
    sys.exit(app.exec())
