"""Main entry point for PokeConsultor."""

import sys
from pokeconsultor.agents.ai_agent import AIAgent
from pokeconsultor.models.llm import LLMRequest
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.service import RAGService
from pokeconsultor.llm.base import llm_profiles


# System message for consistent AI behavior
SYSTEM_MESSAGE = (
    "## ROLE E COMPORTAMENTO\n"
    "Você é um agente de consultoria que responde EXCLUSIVAMENTE com base no contexto fornecido.\n"
    "Você NÃO É um assistente geral de IA. Você é um motor de busca + síntese do RAG.\n"
    "\n"
    "## 🚨 LIMITES ABSOLUTOS - SEM EXCEÇÕES\n"
    "⛔ PROIBIDO USAR:\n"
    "• Seu conhecimento pré-existente/treinamento\n"
    "• Informações gerais mesmo que 'óbvias'\n"
    "• Senso comum ou conhecimento do mundo\n"
    "\n"
    "✅ PERMITIDO E INCENTIVADO:\n"
    "• **Inferência Lógica Contextual**: Você DEVE realizar deduções lógicas simples se houver evidências no contexto.\n"
    "   • *Exemplo*: Se o contexto cita que Harry e Gina têm filhos, você pode inferir que são um casal.\n"
    "• **Transparência de Inferência**: SEMPRE que concluir algo por inferência (não dito explicitamente), você DEVE iniciar o trecho com: \"**Apesar de não ser expresso explicitamente no contexto, é possível presumir que...**\" e então explicar a lógica baseada nos fatos fornecidos.\n"
    "• Sintetizar e reorganizar informações do contexto.\n"
    "• Cite explicitamente as fontes (ex: '(Fonte: nome.pdf, pág. X)').\n"
    "\n"
    "## 🎯 ESTRATÉGIA DE RESPOSTA\n"
    "1️⃣ Receba a pergunta\n"
    "2️⃣ Procure no contexto fornecido (incluindo o que pode ser inferido logicamente)\n"
    "3️⃣ SE encontrar → Sintetize UMA RESPOSTA COMPLETA\n"
    "   • Use a frase de transparência se for uma inferência.\n"
    "4️⃣ SE NÃO encontrar → Responda APENAS: 'Não tenho essa informação no contexto fornecido.'\n"
    "\n"
    "## TESTE DE VERDADE\n"
    "Antes de responder, faça estas perguntas:\n"
    "• Esta informação está no contexto (explícita ou implicitamente)? SIM → Responda | NÃO → 'Não tenho essa informação'\n"
    "• Se for uma inferência, usei a frase obrigatória de transparência? SIM → OK | NÃO → Adicione a frase\n"
    "• Estou usando algo fora do contexto? SIM → APAGUE | NÃO → Continue\n"
    "\n"
    "## ESTILO\n"
    "- Claro, estruturado e sem jargões desnecessários\n"
    "- Organize por relevância (resposta direta → detalhes → contexto)\n"
    "- Use formatação (negrito, listas) para legibilidade\n"
    "- Sempre cite a fonte do contexto EXATAMENTE como fornecido no cabeçalho [n](Fonte: ...)\n"
)


def print_header():
    """Print the application header."""
    print("\033[1;36m" + "=" * 60)
    print("⚙️  INICIALIZANDO POKECONSULTOR")
    print("=" * 60 + "\033[0m")


def print_ready():
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


def main():
    """Main execution loop."""
    print_header()

    try:
        # 1. Configuration
        llm = llm_profiles.get_profile("default")

        # 2. Initialize RAG service
        print(f"[1] 📂 Carregando RAG service (modelo: {llm.model})...")
        rag_service = RAGService(llm_model=llm.model)

        # 3. Initialize AI Agent
        print(f"[2] 🤖 Inicializando LLM ({llm.provider}:{llm.model})...")
        agent = AIAgent(llm=llm)

        # 4. Setup Agent with RAG
        print("[3] 🎯 Configurando AI Agent com RAG...")

        print_ready()

        debug_mode = False
        use_rag = True

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
                    debug_mode = not debug_mode
                    status = "ATIVADO" if debug_mode else "DESATIVADO"
                    print(f"\n⚙️ Modo debug {status}")
                    continue

                # Check for RAG toggle
                if query.lower() == "rag":
                    use_rag = not use_rag
                    status = "ATIVADO" if use_rag else "DESATIVADO"
                    print(f"\n📚 Uso de RAG {status}")
                    continue

                # Memory commands
                if query.lower() in ["memória", "memory"]:
                    history = agent.get_history()
                    if not history:
                        print("\n🧠 Memória vazia.")
                    else:
                        print("\n" + "=" * 80)
                        print(f"📋 HISTÓRICO COMPLETO DE CONVERSA ({len(history)} mensagens)")
                        print("=" * 80)
                        
                        pair_idx = 1
                        for i in range(0, len(history), 2):
                            user_msg = history[i]
                            print(f"\n[Pergunta {pair_idx}] 👤")
                            print("-" * 80)
                            print(f"❓ {user_msg['content']}")
                            
                            if i + 1 < len(history):
                                assistant_msg = history[i+1]
                                print(f"\n[Resposta {pair_idx}] 🤖")
                                print("-" * 80)
                                print(f"✨ {assistant_msg['content']}")
                            pair_idx += 1
                        print("\n" + "=" * 80)
                    continue

                if query.lower() in ["limpar_memória", "clear_memory"]:
                    agent.clear_memory()
                    print("\n🧠 Memória limpa!")
                    continue

                print("\n" + "=" * 60)
                print(f"🔍 QUERY: {query}")
                print("=" * 60)

                print("\n[AI] 🤖 Gerando resposta...")

                # RAG Process
                rag_results = []
                retrieved_context = ""
                used_indices = []

                if use_rag:
                    # Retrieve documents
                    rag_results = rag_service.retrieve(query)
                    
                    if rag_results:
                        print(f"✅ {len(rag_results)} documentos recuperados de fontes locais.")
                        # Format context for prompt
                        retrieved_context = rag_service.format_results(rag_results)
                        
                        # Identify which results are in the context (more robustly)
                        cleaned_context = " ".join(retrieved_context.split())
                        for i, doc in enumerate(rag_results, 1):
                            # Use first 100 chars, normalized whitespace
                            check_text = " ".join(doc.page_content[:100].split())
                            if check_text and check_text in cleaned_context:
                                used_indices.append(i)

                # Generate response
                request = LLMRequest(
                    prompt=query,
                    system_message=SYSTEM_MESSAGE,
                    context=retrieved_context if use_rag else None
                )
                
                response_text = agent.respond(request)

                # Print response
                print("\n" + "=" * 60)
                print("✨ RESPOSTA DA IA")
                print("=" * 60)
                print(f"\n{response_text}")

                # Debug info
                if debug_mode:
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
                        approx_tokens = rag_service.count_tokens(retrieved_context)
                        print(
                            f"Total enviado: {sent_count} de {len(rag_results)} documentos "
                            f"| {sent_chars} chars | ~{approx_tokens} tokens"
                        )
                        print("-" * 80)
                        print(retrieved_context)
                    else:
                        print("Nenhum contexto enviado.")

                print("-" * 60)
                print("-" * 60)

            except KeyboardInterrupt:
                print("\n⚠️ Operação interrompida pelo usuário.")
                continue

    except Exception as e:
        logger.exception("Erro fatal na aplicação")
        print(f"\n\033[1;31m❌ ERRO FATAL: {e}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
