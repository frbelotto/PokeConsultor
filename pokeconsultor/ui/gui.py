"""PySide6 GUI for PokeConsultor."""

import sys
import threading

from langchain.messages import HumanMessage
from pokeconsultor.services.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from PySide6.QtCore import QObject, Qt, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pokeconsultor.llm.prompts import SYSTEM_MESSAGE


class WorkerSignals(QObject):
    """Signals for the background worker."""

    response_ready = Signal(str, list, str, list)
    error_occurred = Signal(str)


class PokeConsultorGUI(QMainWindow):
    """PySide6 interface for PokeConsultor."""

    def __init__(self, agent, rag_service):
        super().__init__()
        self.agent = agent
        self.rag_service = rag_service
        self.signals = WorkerSignals()
        self.signals.response_ready.connect(self._on_response_ready)
        self.signals.error_occurred.connect(self._on_error)
        self.counter: int = 0

        self.init_ui()

    def init_ui(self):
        """Initialize the GUI layout."""
        self.setWindowTitle("PokeConsultor - AI Assistant")
        self.resize(1000, 700)

        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Splitter for Chat and Debug
        self.splitter = QSplitter(Qt.Horizontal)

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

        self.side_label = QLabel("🔍 Informações de Contexto / Debug")
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

        self.rag_checkbox = QCheckBox("Usar RAG")
        self.rag_checkbox.setChecked(True)
        toolbar_layout.addWidget(self.rag_checkbox)

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
        self.help_button.setCursor(Qt.PointingHandCursor)
        self.help_button.setStyleSheet(button_style)

        help_menu = QMenu(self)
        help_menu.addAction("Manual (README)", self.show_help)
        help_menu.addAction("Código Fonte (GitHub)", self.open_repo)
        self.help_button.setMenu(help_menu)
        toolbar_layout.addWidget(self.help_button)

        # Clear Memory Button
        self.clear_mem_button = QPushButton("Limpar Memória")
        self.clear_mem_button.setCursor(Qt.PointingHandCursor)
        self.clear_mem_button.setStyleSheet(button_style)
        self.clear_mem_button.clicked.connect(self.clear_memory)
        toolbar_layout.addWidget(self.clear_mem_button)

        main_layout.addLayout(toolbar_layout)

        # Initial message
        self._append_chat(
            "🤖 <b>Sistema pronto!</b> Faça suas perguntas sobre Pokémon.", "system"
        )

    def show_help(self):
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

    def open_repo(self):
        """Open the repository URL in the default browser."""
        QDesktopServices.openUrl(QUrl("https://github.com/frbelotto/PokeConsultor"))

    def _toggle_debug_panel(self, state):
        """Show/hide the debug side panel."""
        self.side_panel.setVisible(state == Qt.Checked.value)

    def _append_chat(self, text, sender):
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
        self.chat_display.moveCursor(QTextCursor.End)

    def send_message(self):
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

    def _process_query(self, query):
        """Background thread to process the query."""
        try:
            use_rag = self.rag_checkbox.isChecked()

            rag_results = []
            retrieved_context = ""
            used_indices = []

            if use_rag:
                rag_results = self.rag_service.retrieve(query)
                if rag_results:
                    retrieved_context = self.rag_service.format_results(rag_results)
                    cleaned_context = " ".join(retrieved_context.split())
                    for i, doc in enumerate(rag_results, 1):
                        check_text = " ".join(doc.page_content[:100].split())
                        if check_text and check_text in cleaned_context:
                            used_indices.append(i)

                            request = HumanMessage(content=query)
                logger.info(f"User prompt: {request}")
                
            ragcontext = HumanMessage(content="")  
            if use_rag:
                ragcontext = HumanMessage(
                    content="Para responder a questão, saiba que o contexto relevante é: "
                    + retrieved_context
                )

            response_text = self.agent.respond(prompt=request, ragcontext=ragcontext)

            # Garantir que a resposta seja string
            if not isinstance(response_text, str):
                try:
                    response_text = str(response_text.content)
                except Exception as e:
                    response_text = f"[ERRO ao converter resposta da IA: {e}]"

            # Format debug info
            self.debug_info = self._format_debug_info(
                rag_results, retrieved_context, used_indices
            )

            self.signals.response_ready.emit(
                response_text, rag_results, retrieved_context, used_indices
            )

        except Exception as e:
            self.signals.error_occurred.emit(str(e))

    def _on_response_ready(
        self, response_text, rag_results, retrieved_context, used_indices
    ):
        """Handle UI update when response is ready."""
        self._append_chat(response_text, "ai")

        # Update debug panel
        debug_text = self._format_debug_info(
            rag_results, retrieved_context, used_indices
        )
        self.debug_display.setPlainText(debug_text)

        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        self.input_field.setFocus()

    def _on_error(self, error_msg):
        """Handle errors during processing."""
        self._append_chat(f"❌ Erro: {error_msg}", "system")
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

    def _format_debug_info(self, rag_results, retrieved_context, used_indices):
        """Format debug information for the side panel."""
        output = []
        output.append("=== STAGE 1: DOCUMENTOS RECUPERADOS ===")
        output.append(f"Total: {len(rag_results)}\n")

        for i, doc in enumerate(rag_results, 1):
            filename = doc.metadata.get("file_path", "unknown").split("/")[-1]
            page = doc.metadata.get("page_number")
            row = doc.metadata.get("row_number")
            ref = filename
            if page:
                ref += f" (pág. {page})"
            elif row:
                ref += f" (linha {row})"

            in_context = " [SENT]" if i in used_indices else ""
            output.append(f"[{i:2d}]{in_context} {ref}")
            output.append(f"   {doc.page_content[:100]}...\n")

        output.append("\n=== STAGE 2: CONTEXTO ENVIADO ===")
        if retrieved_context:
            approx_tokens = self.rag_service.count_tokens(retrieved_context)
            output.append(f"Tokens aprox: {approx_tokens}")
            output.append("-" * 40)
            output.append(retrieved_context)
        else:
            output.append("Nenhum contexto enviado.")

        return "\n".join(output)

    def clear_memory(self):
        """Clear the AI agent memory and UI displays."""
        self.agent.memory.clear()
        self.chat_display.clear()
        self.debug_display.clear()
        self._append_chat("🧠 Memória e histórico limpos!", "system")


def run_gui(agent, rag_service):
    """Entry point for the GUI version."""
    app = QApplication(sys.argv)
    window = PokeConsultorGUI(agent, rag_service)
    window.show()
    sys.exit(app.exec())
