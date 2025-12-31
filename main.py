"""PokeConsultor - Interactive AI Pokemon Consultant."""

import sys

from pokeconsultor.agents.ai_agent import AIAgent
from pokeconsultor.llm.groq_llm import GroqLLM


def main() -> None:
    """Run the PokeConsultor interactive agent."""
    try:
        # Initialize LLM and Agent
        print("⚙️  Inicializando PokeConsultor...")
        llm = GroqLLM()
        agent = AIAgent(llm=llm)
        print("✅ Pronto!\n")

    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        print("\n💡 Verifique se:")
        print("   - O arquivo .env existe e está configurado")
        print("   - A variável LLM_API_KEY está definida")
        print("   - O ambiente virtual está ativado")
        sys.exit(1)

    # Interactive mode
    print("=" * 60)
    print("🎮 PokeConsultor - Consultor de Pokémon com IA")
    print("=" * 60)
    print("\n💬 Faça suas perguntas sobre Pokémon!")
    print("📝 Comandos disponíveis:")
    print("   - 'sair' ou 'exit' para encerrar")
    print("   - 'limpar' ou 'clear' para limpar o console")
    print("   - Ctrl+C para interromper\n")

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

                # Skip empty queries
                continue

            # Process query
            print("\n⏳ Consultando...")
            response = agent.consult(prompt=query)

            print(f"\n✨ Resposta:\n{response}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Encerrando...")
            break

        except Exception as e:
            print(f"\n❌ Erro ao processar consulta: {e}\n")
            print("💡 Tente reformular sua pergunta ou verifique a conexão.\n")


if __name__ == "__main__":
    main()
