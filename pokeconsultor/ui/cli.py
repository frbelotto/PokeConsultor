"""CLI UI for PokeConsultor."""

import sys
from typing import Any

from langchain.messages import HumanMessage

from pokeconsultor.services.logger import logger


class PokeConsultorCLI:
    """Terminal-based interface for PokeConsultor."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.debug_mode = False

    def print_header(self) -> None:
        """Print the application header."""
        print("\033[1;36m" + "=" * 60)
        print("⚙️  INICIALIZANDO POKECONSULTOR")
        print("=" * 60 + "\033[0m")

    def print_ready(self) -> None:
        """Print system ready message."""
        print("\n\033[1;32m✅ Sistema pronto para consultas!\033[0m")
        print("\n" + "=" * 60)
        print("\033[1;34m🎮 POKECONSULTOR - CONSULTOR DE POKÉMON COM IA\033[0m")
        print("=" * 60)
        print("\n💬 Faça suas perguntas sobre Pokémon!")
        print("\n📝 Comandos disponíveis:")
        print("   • 'sair' ou 'exit' para encerrar")
        print("   • 'limpar' ou 'clear' para limpar o console")
        print("   • 'debug' para ativar/desativar modo debug")
        print("   • 'memória' ou 'memory' para ver histórico de conversas")
        print("   • 'limpar_memória' ou 'clear_memory' para apagar histórico")
        print("   • Ctrl+C para interromper")

    def run(self) -> None:
        """Main execution loop for CLI."""
        self.print_ready()

        while True:
            try:
                print("\n\033[1;33m🔍 Sua pergunta: \033[0m", end="", flush=True)
                query = sys.stdin.readline().strip()

                if not query:
                    continue

                # Check for exit commands
                if query.lower() in ["sair", "exit"]:
                    print("\n👋 Até logo!")
                    break

                # Check for clear commands
                if query.lower() in ["limpar", "clear", "cls"]:
                    print("\033[2J\033[H")  # Clear console
                    continue

                # Check for debug toggle
                if query.lower() == "debug":
                    self.debug_mode = not self.debug_mode
                    status = "ATIVADO" if self.debug_mode else "DESATIVADO"
                    print(f"\n⚙️ Modo debug {status}")
                    continue

                # Memory commands
                if query.lower() in ["memória", "memory"]:
                    history = self.agent.get_state_history()
                    if not history:
                        print("\n🧠 Memória vazia.")
                    else:
                        print("\n" + "=" * 80)
                        print("📋 HISTÓRICO COMPLETO DE CONVERSA ")
                        print("=" * 80)

                        self._print_memory_history_raw(history)

                        print("\n" + "=" * 80)
                    continue

                if query.lower() in ["limpar_memória", "clear_memory"]:
                    self.agent.clear_thread_memory()
                    print("\033[2J\033[H")  # Clear terminal
                    print("\n🧠 Memória e terminal limpos!")
                    continue

                print("\n" + "=" * 60)
                print(f"🔍 QUERY: {query}")
                print("=" * 60)

                print("\n[AI] 🤖 Gerando resposta...")

                # Build a single HumanMessage for the agent (agent.respond expects a HumanMessage)
                request = HumanMessage(content=query)
                logger.info(f"User prompt: {request}")

                response_text = self.agent.respond(prompt=request)

                # Print response
                print("\n" + "=" * 60)
                print("✨ RESPOSTA DA IA")
                print("=" * 60)
                print(f"\n{response_text}")

                # Debug info
                if self.debug_mode:
                    self._print_debug_info()

                print("-" * 60)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n⚠️ Operação interrompida pelo usuário.")
                continue

    def _print_debug_info(self) -> None:
        """Print debug information from graph state history."""
        history = self.agent.get_state_history()
        tool_usage = self.agent.get_latest_interaction_tool_usage()

        print("\n" + "🔍" * 30)
        print("[DEBUG] 📚 ESTADO RECENTE DO AGENTE")
        print("🔍" * 30)

        used = "SIM" if tool_usage["used"] else "NÃO"
        names = ", ".join(tool_usage["tool_names"]) if tool_usage["tool_names"] else "-"
        print(f"Tool usada na última interação: {used}")
        print(f"Tools: {names}")

        if not history:
            print("Nenhum histórico disponível.")
            return

        print(f"Total de estados: {len(history)}")
        print("-" * 80)
        for state in history[:3]:
            print(state)

    def _print_memory_history_raw(self, history: list[Any]) -> None:
        """Print raw conversation history snapshots for deep debugging."""
        from pprint import PrettyPrinter

        printer = PrettyPrinter(width=140, compact=False, sort_dicts=False)

        for index, snapshot in enumerate(history, 1):
            values = getattr(snapshot, "values", {})
            config = getattr(snapshot, "config", {})
            metadata = getattr(snapshot, "metadata", {})
            created_at = getattr(snapshot, "created_at", None)

            messages = values.get("messages", []) if isinstance(values, dict) else []
            formatted_messages = [
                self._format_message_for_display(message) for message in messages
            ]

            state_payload = {
                "state_index": index,
                "created_at": created_at,
                "metadata": metadata,
                "configurable": config.get("configurable", {})
                if isinstance(config, dict)
                else {},
                "messages": formatted_messages,
            }

            print("\n" + "-" * 80)
            printer.pprint(state_payload)

    @staticmethod
    def _format_message_for_display(message: Any) -> dict[str, Any]:
        """Convert LangChain message objects into readable dictionaries."""
        payload: dict[str, Any] = {
            "type": type(message).__name__,
            "content": getattr(message, "content", ""),
        }

        name = getattr(message, "name", None)
        if name:
            payload["name"] = name

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = tool_calls

        additional_kwargs = getattr(message, "additional_kwargs", None)
        if additional_kwargs:
            payload["additional_kwargs"] = additional_kwargs

        response_metadata = getattr(message, "response_metadata", None)
        if response_metadata:
            payload["response_metadata"] = response_metadata

        usage_metadata = getattr(message, "usage_metadata", None)
        if usage_metadata:
            payload["usage_metadata"] = usage_metadata

        return payload
