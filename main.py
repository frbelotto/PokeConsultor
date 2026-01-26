"""Main entry point for PokeConsultor."""

import argparse
import sys
from pokeconsultor.agents.ai_agent import AIAgent
from pokeconsultor.services.logger import logger
from pokeconsultor.services.rag.service import RAGService
from pokeconsultor.llm.base import llm_profiles
from pokeconsultor.ui.cli import PokeConsultorCLI


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PokeConsultor - AI Assistant")
    parser.add_argument("--gui", action="store_true", help="Run with PySide6 GUI")
    args = parser.parse_args()

    try:
        # 1. Configuration
        llm = llm_profiles.get_profile("default")

        # 2. Initialize modules
        print(f"⚙️  Inicializando PokeConsultor (modelo: {llm.model})...")

        rag_service = RAGService(llm_model=llm.model)
        agent = AIAgent(llm=llm)

        if args.gui:
            try:
                from pokeconsultor.ui.gui import run_gui

                run_gui(agent, rag_service)
            except ImportError as e:
                print(f"\n❌ Erro ao carregar GUI: {e}")
                print("Certifique-se de que o PySide6 está instalado.")
                sys.exit(1)
        else:
            cli = PokeConsultorCLI(agent, rag_service)
            cli.print_header()
            cli.run()

    except KeyboardInterrupt:
        print("\n👋 Até logo!")
    except Exception as e:
        logger.exception("Erro fatal na aplicação")
        print(f"\n\033[1;31m❌ ERRO FATAL: {e}\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
