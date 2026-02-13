"""CLI UI for PokeConsultor."""

import sys

from langchain.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pokeconsultor.services.logger import logger


class PokeConsultorCLI:
    """Terminal-based interface for PokeConsultor."""

    def __init__(self, agent, rag_service):
        self.agent = agent
        self.rag_service = rag_service
        self.debug_mode = False

    def print_header(self):
        """Print the application header."""
        print("\033[1;36m" + "=" * 60)
        print("⚙️  INICIALIZANDO POKECONSULTOR")
        print("=" * 60 + "\033[0m")

    def print_ready(self):
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

    def run(self):
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
                    # history = self.agent.memory.get_history()
                    history = self.agent._agent.get_state_history(
                        {"configurable": {"thread_id": str(self.agent._threadid)}}
                    )
                    if not history:
                        print("\n🧠 Memória vazia.")
                    else:
                        print("\n" + "=" * 80)
                        print(f"📋 HISTÓRICO COMPLETO DE CONVERSA ")
                        print("=" * 80)

                        for message in history:
                            print(message)

                        print("\n" + "=" * 80)
                    continue

                if query.lower() in ["limpar_memória", "clear_memory"]:
                    # self.agent.memory.clear()
                    self.agent._agent.checkpointer.delete_thread(
                        str(self.agent._threadid)
                    )
                    print("\033[2J\033[H")  # Clear terminal
                    print("\n🧠 Memória e terminal limpos!")
                    continue

                print("\n" + "=" * 60)
                print(f"🔍 QUERY: {query}")
                print("=" * 60)

                print("\n[AI] 🤖 Gerando resposta...")

                # Build the user message
                # The agent will decide whether to call retrieve_context tool
                request = HumanMessage(content=query)
                logger.info(f"User prompt: {request}")

                # No manual RAG - the agent will use the tool if needed
                response_text = self.agent.respond(
                    prompt=request, ragcontext=HumanMessage(content="")
                )

                # Print response
                print("\n" + "=" * 60)
                print("✨ RESPOSTA DA IA")
                print("=" * 60)
                print(f"\n{response_text}")

                # Debug info (simplified - tool calls are logged automatically)
                if self.debug_mode:
                    print("\n" + "🔍" * 30)
                    print("[DEBUG] 🔧 Tool-calling mode ativo")
                    print("Verifique os logs para detalhes de chamadas de ferramentas")
                    print("🔍" * 30)

                print("-" * 60)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n⚠️ Operação interrompida pelo usuário.")
                continue
