import sys

from pokeconsultor.agents.ai_agent import AIAgent
from pokeconsultor.config import settings
from pokeconsultor.llm.base import LLMProfiles, llm_profiles
from pokeconsultor.models.llm import LLMRequest
from pokeconsultor.services.rag import RAGService


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
        # "Você é um especialista em Pokémon. Responda APENAS baseado nas "
        "Você é um agente de IA prestando consultoria sobre todos os assuntos que localizar em sua base de conhecimentos. "
        "Responda APENAS com base nas informações fornecidas no contexto. Se a informação não estiver no contexto, diga que não tem dados sobre o assunto. "
        "Seja claro, objetivo e amigável."
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
    print("   • 'memória' ou 'memory' para ver histórico de conversas")
    print("   • 'limpar_memória' ou 'clear_memory' para apagar histórico")
    print("   • Ctrl+C para interromper\n")

    debug_mode = False

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

            # Show RAG retrieval in debug mode
            retrieved_context: str | None = None
            if debug_mode:
                print("\n" + "🔍" * 30)
                print("[RAG] 📚 Recuperando documentos relevantes da base de dados...")
                print("🔍" * 30)
                print("-" * 60)
                try:
                    results = rag_service.retrieve(query, k=3)
                    print(
                        f"[RAG] ✓ Encontrados {len(results)} resultados (k=3, rerank_k=3)\n"
                    )

                    # Show all retrieved results before formatting
                    print("[RAG] 📋 DOCUMENTOS RECUPERADOS DA BASE:")
                    print("=" * 60)
                    for i, (doc, score) in enumerate(results, 1):
                        doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                        doc_tokens = rag_service._count_tokens(doc)  # noqa: SLF001
                        relevance_bar = "█" * int(score * 10) + "░" * (
                            10 - int(score * 10)
                        )
                        print(f"\n  📄 Documento {i}")
                        print(
                            f"     Score: {score:.4f} [{relevance_bar}] ~{doc_tokens} tokens"
                        )
                        print(f"     └─ {doc_preview}\n")

                    retrieved_context = rag_service.format_results(results)

                    # Show what was actually sent to LLM
                    print("\n" + "=" * 60)
                    print("[LLM] 📤 CONTEXTO PREPARADO PARA ENVIAR À LLM:")
                    print("=" * 60)
                    if retrieved_context:
                        included = [
                            part.split("\n", 1)[0]
                            for part in retrieved_context.split("\n\n")
                            if part.startswith("[Resultado ")
                        ]
                        approx_tokens = rag_service._count_tokens(retrieved_context)  # noqa: SLF001
                        print(
                            f"\n✓ Incluídos: {len(included)} de {len(results)} documentos "
                            f"({', '.join(included)})"
                        )
                        print(f"✓ Tokens utilizados: ~{approx_tokens} \n")
                        print("-" * 60)
                        print(retrieved_context)
                        print("-" * 60)
                    else:
                        print("(nenhum contexto gerado)")
                    print("=" * 60 + "\n")
                except Exception as e:
                    print(f"⚠️  Erro ao recuperar documentos: {e}")
                    print("-" * 60)

            print("\n[AI] 🤖 Gerando resposta...\n")
            request = LLMRequest(
                prompt=query,
                system_message=system_message,
                context=retrieved_context,
            )
            response = ai_agent.respond(request)

            print("\n" + "=" * 60)
            print("✨ RESPOSTA DA IA")
            print("=" * 60)
            print(f"\n{response}\n")

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
