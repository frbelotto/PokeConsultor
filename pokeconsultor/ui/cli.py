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
        self.use_rag = True

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
        print("   • 'rag' para ativar/desativar uso de contexto RAG")
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

                # Check for RAG toggle
                if query.lower() == "rag":
                    self.use_rag = not self.use_rag
                    status = "ATIVADO" if self.use_rag else "DESATIVADO"
                    print(f"\n📚 Uso de RAG {status}")
                    continue

                # Memory commands
                if query.lower() in ["memória", "memory"]:
                    history = self.agent.memory.get_history()
                    if not history:
                        print("\n🧠 Memória vazia.")
                    else:
                        print("\n" + "=" * 80)
                        print(
                            f"📋 HISTÓRICO COMPLETO DE CONVERSA ({len(history)} mensagens)"
                        )
                        print("=" * 80)

                        pair_idx = 1
                        for i in range(0, len(history), 2):
                            user_msg = history[i]
                            print(f"\n[Pergunta {pair_idx}] 👤")
                            print("-" * 80)
                            print(f"❓ {user_msg['content']}")

                            if i + 1 < len(history):
                                assistant_msg = history[i + 1]
                                print(f"\n[Resposta {pair_idx}] 🤖")
                                print("-" * 80)
                                print(f"✨ {assistant_msg['content']}")
                            pair_idx += 1
                        print("\n" + "=" * 80)
                    continue

                if query.lower() in ["limpar_memória", "clear_memory"]:
                    self.agent.memory.clear()
                    print("\033[2J\033[H")  # Clear terminal
                    print("\n🧠 Memória e terminal limpos!")
                    continue

                print("\n" + "=" * 60)
                print(f"🔍 QUERY: {query}")
                print("=" * 60)

                print("\n[AI] 🤖 Gerando resposta...")

                # RAG Process
                rag_results = []
                retrieved_context = ""
                used_indices = []

                if self.use_rag: 
                    # Retrieve documents
                    rag_results = self.rag_service.retrieve(query)

                    if rag_results:
                        print(
                            f"✅ {len(rag_results)} documentos recuperados de fontes locais."
                        )
                        # Format context for prompt
                        retrieved_context = self.rag_service.format_results(rag_results)

                        # Identify which results are in the context (more robustly)
                        cleaned_context = " ".join(retrieved_context.split())
                        for i, doc in enumerate(rag_results, 1):
                            # Use first 100 chars, normalized whitespace
                            check_text = " ".join(doc.page_content[:100].split())
                            if check_text and check_text in cleaned_context:
                                used_indices.append(i)

                # Build a single HumanMessage for the agent (agent.respond expects a HumanMessage)
                request = HumanMessage(content=query)
                logger.info(f"User prompt: {request}")
                
                ragcontext = HumanMessage(content="")  
                if self.use_rag:
                    ragcontext = HumanMessage(
                        content="Para responder a questão, saiba que o contexto relevante é: "
                        + retrieved_context
                    )

                response_text = self.agent.respond(prompt=request, ragcontext=ragcontext)

                # Print response
                print("\n" + "=" * 60)
                print("✨ RESPOSTA DA IA")
                print("=" * 60)
                print(f"\n{response_text}")

                # Debug info
                if self.debug_mode:
                    self._print_debug_info(rag_results, retrieved_context, used_indices)

                print("-" * 60)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n⚠️ Operação interrompida pelo usuário.")
                continue

    def _print_debug_info(self, rag_results, retrieved_context, used_indices):
        """Print debug information about RAG process."""
        print("\n" + "🔍" * 30)
        print("[RAG] 📚 PIPELINE DE RECUPERAÇÃO E CONTEXTO")
        print("🔍" * 30)

        # Stage 1: All retrieved results
        print("\n[STAGE 1] 📋 DOCUMENTOS RECUPERADOS (retrieve):")
        print("=" * 80)
        print(f"Total recuperado: {len(rag_results)} documentos\n")
        for i, doc in enumerate(rag_results, 1):
            content = doc.page_content
            doc_preview = content[:160] + "..." if len(content) > 160 else content
            doc_chars = len(content)

            filename = doc.metadata.get("file_path", "unknown").split("/")[-1]
            page = doc.metadata.get("page_number")
            row = doc.metadata.get("row_number")

            ref = filename
            if page:
                ref += f" (pág. {page})"
            elif row:
                ref += f" (linha {row})"

            in_context_flag = " ✓ ENVIADO" if i in used_indices else ""
            print(
                f"  [{i:2d}] Fonte: {ref:30s} | {doc_chars:5d} chars{in_context_flag}"
            )
            print(f"       └─ {doc_preview}\n")

        # Stage 2: What was sent to LLM
        print("\n[STAGE 2] 📤 CONTEXTO ENVIADO À LLM (format_results):")
        print("=" * 80)
        if retrieved_context:
            sent_count = len(used_indices)
            sent_chars = len(retrieved_context)
            approx_tokens = self.rag_service.count_tokens(retrieved_context)
            print(
                f"Total enviado: {sent_count} de {len(rag_results)} documentos "
                f"| {sent_chars} chars | ~{approx_tokens} tokens"
            )
            print("-" * 80)
            print(retrieved_context)
        else:
            print("Nenhum contexto enviado.")
