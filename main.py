import sys

from pokeconsultor.agents.ai_agent import AIAgent
from pokeconsultor.config import settings
from pokeconsultor.llm.base import llm_profiles
from pokeconsultor.models.llm import LLMRequest
from pokeconsultor.services.rag.service import RAGService


def main() -> None:
    """Run the PokeConsultor interactive agent."""
    try:
        # Initialize LLM and Agent
        print("\n" + "=" * 60)
        print("⚙️  INICIALIZANDO POKECONSULTOR")
        print("=" * 60)

        print("\n[1] 📂 Carregando RAG service ...")
        rag_service = RAGService(
            use_cache=True,
            llm_model=settings.LLM_DEFAULT_MODEL,
        )

        print(
            f"[2] 🤖 Inicializando LLM ({settings.LLM_DEFAULT_PROVIDER}/{settings.LLM_DEFAULT_MODEL})..."
        )
        print("[3] 🎯 Configurando AI Agent com RAG...")
        ai_agent = AIAgent(llm=llm_profiles.get_profile("default"))
        print("\n✅ Sistema pronto para consultas!\n")

    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        print("\n💡 Verifique se:")
        print("   - O arquivo .env existe e está configurado")
        print("   - A variável LLM_API_KEY está definida")
        print("   - O ambiente virtual está ativado")
        sys.exit(1)

    # System message for consistent AI behavior
    system_message = (
        "## ROLE E COMPORTAMENTO\n"
        "Você é um agente de consultoria que responde perguntas APENAS com base no contexto fornecido.\n"
        "\n"
        "## 🚨 INSTRUÇÃO ABSOLUTA - COMPLIANCE NÃO NEGOCIÁVEL 🚨\n"
        "RESPONDA EXCLUSIVAMENTE COM BASE NO CONTEXTO. SEM EXCEÇÕES.\n"
        "\n"
        "✋ PARAR AQUI - LEIA COM MÁXIMA ATENÇÃO:\n"
        "• SE NÃO HÁ CONTEXTO = VOCÊ NÃO PODE RESPONDER COM INFORMAÇÕES\n"
        "• SE O CONTEXTO NÃO CONTÉM A RESPOSTA = VOCÊ NÃO PODE ADIVINHAR\n"
        "• NUNCA USE CONHECIMENTO DO SEU TREINAMENTO PARA PREENCHER LACUNAS\n"
        "• NUNCA FABRIQUE OU ASSUMA INFORMAÇÕES NÃO EXPLÍCITAS NO CONTEXTO\n"
        "• PROIBIDO ALUCINAR, INVENTAR, DEDUZIR OU USAR SENSO COMUM\n"
        "\n"
        "## COMPORTAMENTO MANDATÓRIO\n"
        "Ao receber uma pergunta:\n"
        "1️⃣ BUSQUE A RESPOSTA APENAS NO CONTEXTO FORNECIDO\n"
        "2️⃣ SE ENCONTRAR = RESPONDA COM REFERÊNCIAS EXPLÍCITAS (ex: '[Resultado 2]')\n"
        "3️⃣ SE NÃO ENCONTRAR = RESPONDA IMEDIATAMENTE: 'Não tenho essa informação no contexto fornecido.'\n"
        "\n"
        "## EXEMPLOS DO QUE FAZER vs O QUE NÃO FAZER\n"
        "✅ CORRETO: Contexto vazio → 'Não tenho essa informação no contexto.'\n"
        "✅ CORRETO: Contexto diz 'X' → Responda apenas sobre 'X'\n"
        "❌ ERRADO: Contexto vazio → [Inventar com conhecimento geral]\n"
        "❌ ERRADO: Contexto diz 'X' → [Adicionar 'Y' do seu conhecimento]\n"
        "\n"
        "## ESTILO DE RESPOSTA\n"
        "- Seja claro, amigável e estruturado.\n"
        "- Responda de forma completa quando apropriado, evitando omissões.\n"
        "- Estruture a saída com seções curtas.\n"
        "- Sempre cite explicitamente os trechos do contexto usado.\n"
        "- Prefira listas com bullets quando útil.\n"
    )

    # Interactive mode
    print("=" * 60)
    print("🎮 POKECONSULTOR - CONSULTOR DE POKÉMON COM IA")
    print("=" * 60)
    print("\n💬 Faça suas perguntas sobre Pokémon!")
    print("\n📝 Comandos disponíveis:")
    print("   • 'sair' ou 'exit' para encerrar")
    print("   • 'limpar' ou 'clear' para limpar o console")
    print("   • 'debug' para ativar/desativar modo debug")
    print("   • 'rag' para ativar/desativar uso de contexto RAG")
    print("   • 'memória' ou 'memory' para ver histórico de conversas")
    print("   • 'limpar_memória' ou 'clear_memory' para apagar histórico")
    print("   • Ctrl+C para interromper\n")

    debug_mode = False
    rag_enabled = True

    while True:
        try:
            query = input("🔍 Sua pergunta: ").strip()

            # Check for exit commands
            if query.lower() in ["sair", "exit", "quit", "q"]:
                print("\n👋 Até a próxima!")
                break

            # Check for clear commands
            if query.lower() in ["limpar", "clear", "cls"]:
                print("\033[2J\033[H")  # Clear console
                continue

            # Check for debug toggle
            if query.lower() == "debug":
                debug_mode = not debug_mode
                status = "ativado" if debug_mode else "desativado"
                print(f"\n🐛 Modo debug {status}\n")
                continue

            # Check for RAG toggle
            if query.lower() == "rag":
                rag_enabled = not rag_enabled
                status = "ativado" if rag_enabled else "desativado"
                print(f"\n📚 RAG {status}\n")
                continue

            # Check for memory commands
            if query.lower() in ["memória", "memory"]:
                summary = ai_agent.memory.get_summary()
                messages = ai_agent.memory.get_messages()
                print(f"\n📚 {summary}\n")
                if messages:
                    print("=" * 80)
                    print("📋 HISTÓRICO COMPLETO DE CONVERSAS")
                    print("=" * 80)

                    # Group messages by user-assistant pairs
                    i = 1
                    msg_idx = 0
                    while msg_idx < len(messages):
                        msg = messages[msg_idx]

                        # Display user message
                        if msg.role.value == "user":
                            timestamp = msg.timestamp.strftime("%H:%M:%S")
                            print(f"\n[Pergunta {i}] 👤 ({timestamp})")
                            print("-" * 80)
                            print(f"❓ {msg.content}")

                            # Look for corresponding assistant response
                            if (
                                msg_idx + 1 < len(messages)
                                and messages[msg_idx + 1].role.value == "assistant"
                            ):
                                response_msg = messages[msg_idx + 1]
                                response_timestamp = response_msg.timestamp.strftime(
                                    "%H:%M:%S"
                                )
                                print(f"\n[Resposta {i}] 🤖 ({response_timestamp})")
                                print("-" * 80)
                                response_preview = response_msg.content
                                if len(response_preview) > 300:
                                    response_preview = (
                                        response_preview[:300]
                                        + "\n\n... [resposta truncada para visualização]"
                                    )
                                print(f"✅ {response_preview}")
                                msg_idx += 1

                            i += 1

                        msg_idx += 1

                    print("\n" + "=" * 80 + "\n")
                continue

            # Check for clear memory command
            if query.lower() in ["limpar_memória", "clear_memory"]:
                ai_agent.memory.clear()
                print("\n✨ Histórico de conversas apagado!\n")
                continue

            # Process query with visual feedback
            print("\n" + "=" * 60)
            print(f"🔍 QUERY: {query}")
            print("=" * 60)

            # RAG retrieval (always executed when enabled; debug shows detalhes)
            retrieved_context: str | None = None
            rag_results: list[tuple[str, float]] = []
            rag_consulted = rag_enabled
            if rag_enabled:
                try:
                    rag_results = rag_service.retrieve(query)
                    retrieved_context = rag_service.format_results(rag_results)

                    # Identify which results were actually sent to LLM
                    used_indices: set[int] = set()
                    if retrieved_context:
                        for line in retrieved_context.splitlines():
                            if line.startswith("[Resultado "):
                                try:
                                    idx_str = line.split("[Resultado ", 1)[1].split(
                                        "]", 1
                                    )[0]
                                    used_indices.add(int(idx_str))
                                except (IndexError, ValueError):
                                    continue

                    if debug_mode:
                        print("\n" + "🔍" * 30)
                        print("[RAG] 📚 PIPELINE DE RECUPERAÇÃO E CONTEXTO")
                        print("🔍" * 30)

                        # Stage 1: All retrieved results
                        print("\n[STAGE 1] 📋 DOCUMENTOS RECUPERADOS (retrieve):")
                        print("=" * 80)
                        print(f"Total recuperado: {len(rag_results)} documentos\n")
                        for i, (doc, score) in enumerate(rag_results, 1):
                            doc_preview = doc[:160] + "..." if len(doc) > 160 else doc
                            doc_chars = len(doc)
                            relevance_bar = "█" * int(score * 10) + "░" * (
                                10 - int(score * 10)
                            )
                            in_context_flag = " ✓ ENVIADO" if i in used_indices else ""
                            print(
                                f"  [{i:2d}] Score: {score:.4f} [{relevance_bar}] {doc_chars:5d} chars{in_context_flag}"
                            )
                            print(f"       └─ {doc_preview}\n")

                        # Stage 2: What was sent to LLM
                        print("\n[STAGE 2] 📤 CONTEXTO ENVIADO À LLM (format_results):")
                        print("=" * 80)
                        if retrieved_context:
                            sent_count = len(used_indices)
                            sent_chars = len(retrieved_context)
                            approx_tokens = rag_service.count_tokens(retrieved_context)
                            print(
                                f"Total enviado: {sent_count} de {len(rag_results)} documentos "
                                f"({sent_chars} chars, ~{approx_tokens} tokens)\n"
                            )
                            print("-" * 80)
                            print(retrieved_context)
                            print("-" * 80)
                        else:
                            print(
                                "(nenhum contexto gerado - limite de tokens atingido ou sem resultados)"
                            )
                        print("=" * 80 + "\n")

                except Exception as e:
                    print(f"⚠️  Erro ao recuperar documentos: {e}")
                    print("-" * 60)
                    rag_consulted = False

            print("\n[AI] 🤖 Gerando resposta...\n")
            request = LLMRequest(
                prompt=query,
                system_message=system_message,
                context=retrieved_context,
            )
            response = ai_agent.respond(request)

            # RAG usage summary (always show)
            consulted_chars = (
                sum(len(doc) for doc, _score in rag_results)
                if rag_consulted and rag_results
                else 0
            )
            sent_chars = len(retrieved_context) if retrieved_context else 0
            rag_sent = bool(retrieved_context)

            print("\n" + "=" * 60)
            print("✨ RESPOSTA DA IA")
            print("=" * 60)
            print(f"\n{response}\n")

            print("[RAG] 📊 Uso nesta consulta:")
            print(f"   • Consultado: {'Sim' if rag_consulted else 'Não'}")
            print(
                f"   • Documentos recuperados: {len(rag_results) if rag_consulted else 0} | Tamanho bruto: {consulted_chars} chars"
            )
            print(
                f"   • Contexto enviado à LLM: {'Sim' if rag_sent else 'Não'} | Tamanho enviado: {sent_chars} chars"
            )
            print("-" * 60)

            # Show additional debug info about the response
            if debug_mode:
                print("\n" + "-" * 60)
                print("[DEBUG] 📊 INFORMAÇÕES DA RESPOSTA:")
                print("-" * 60)
                print(f"✓ Pergunta: {query}")
                print(
                    f"✓ Contexto utilizado: {'Sim (RAG)' if retrieved_context else 'Não'}"
                )
                print("-" * 60)

            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Encerrando...")
            break

        except Exception as e:
            print(f"\n❌ Erro ao processar consulta: {e}\n")
            print("💡 Tente reformular sua pergunta ou verifique a conexão.\n")


if __name__ == "__main__":
    main()
