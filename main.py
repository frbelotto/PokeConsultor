"""Main entry point for PokeConsultor."""

import argparse
import sys

from pokeconsultor.config import settings
from pokeconsultor.services.logger import logger


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PokeConsultor - AI Assistant")
    parser.add_argument("--gui", action="store_true", help="Run with PySide6 GUI")
    args = parser.parse_args()

    settings.export_runtime_env()

    from pokeconsultor.agents.ai_agent import AIAgent
    from pokeconsultor.services.rag.service import RAGService
    from pokeconsultor.services.rag.tooling import build_rag_context_tool
    from pokeconsultor.llm.base import llm_profiles
    from pokeconsultor.ui.cli import PokeConsultorCLI
    from pokeconsultor.llm.prompts import SYSTEM_MESSAGE

    try:
        # 1. Configuration
        llm = llm_profiles.get_profile("default")

        # 2. Initialize modules
        print(f"⚙️  Inicializando PokeConsultor (modelo: {llm.model})...")

        rag_service = RAGService(llm_model=llm.model)
        rag_tool = build_rag_context_tool(rag_service)
        agent = AIAgent(llm=llm, systemprompt=SYSTEM_MESSAGE, tools=[rag_tool])

        if args.gui:
            try:
                from pokeconsultor.ui.gui import run_gui

                run_gui(agent)
            except ImportError as e:
                print(f"\n❌ Erro ao carregar GUI: {e}")
                print("Certifique-se de que o PySide6 está instalado.")
                sys.exit(1)
        else:
            cli = PokeConsultorCLI(agent)
            cli.print_header()
            cli.run()

    except KeyboardInterrupt:
        print("\n👋 Até logo!")
    except Exception as e:
        logger.exception("Erro fatal na aplicação")
        print(f"\n\033[1;31m❌ ERRO FATAL: {e}\033[0m")


if __name__ == "__main__":
    main()
