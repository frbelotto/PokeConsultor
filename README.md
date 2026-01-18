# PokeConsultor

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

An AI-powered intelligent consultant system that leverages Retrieval-Augmented Generation (RAG) to answer questions based on custom knowledge bases. While originally designed for Pokémon data, it can be easily adapted for any domain.

## 🌟 Features

- **🧠 RAG-Powered Responses**: Combines semantic search with Large Language Models (LLMs) for accurate, context-aware answers
- **📚 Multi-Format Support**: Load data from CSV, TXT, PDF, Markdown, and other formats
- **💾 Smart Caching**: Persistent FAISS vector store with automatic cache invalidation
- **🔄 Conversation Memory**: Maintains multi-turn conversation context
- **🎯 Multiple LLM Profiles**: Configure different models for various purposes (executor, supervisor, default)
- **🔌 Provider Agnostic**: Works with Groq, OpenAI, HuggingFace, and other LangChain-supported providers
- **📊 Automatic Context Management**: Dynamically adjusts retrieval based on model context windows
- **🏭 Factory Pattern Loaders**: Automatic file type detection and appropriate loader selection

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) for dependency management (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/frbelotto/PokeConsultor.git
cd PokeConsultor
```

2. **Set up environment with uv**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

3. **Configure environment variables**

Create a `.env` file in the project root:

```env
# LLM Provider API Keys
GROQ_API_KEY=your_groq_api_key_here
# HUGGINGFACE_HUB_TOKEN=your_token_here  # Optional

# Default LLM Configuration
LLM_DEFAULT_PROVIDER=groq
LLM_DEFAULT_MODEL=llama-3.1-8b-instant
LLM_DEFAULT_TEMPERATURE=0.7
LLM_DEFAULT_MAX_TOKENS=500

# Executor Profile
LLM_PROFILE_EXECUTOR_PROVIDER=groq
LLM_PROFILE_EXECUTOR_MODEL=llama-3.1-8b-instant
LLM_PROFILE_EXECUTOR_TEMPERATURE=0.5
LLM_PROFILE_EXECUTOR_MAX_TOKENS=1000

# Supervisor Profile
LLM_PROFILE_SUPERVISOR_PROVIDER=groq
LLM_PROFILE_SUPERVISOR_MODEL=llama-3.1-70b-versatile
LLM_PROFILE_SUPERVISOR_TEMPERATURE=0.3
LLM_PROFILE_SUPERVISOR_MAX_TOKENS=2000

# Application Settings
DATA_PATH=data
CACHE_DIR=.cache/vector_stores
LOG_LEVEL=INFO

# Optional: PokeAPI MCP Server
POKEAPI_MCP_SERVER_URL=http://localhost:3000
POKEAPI_MCP_ENABLED=false
```

4. **Add your data files**

Place your knowledge base files in the `data/` directory:
- CSV files (e.g., `Treinadores.csv`)
- Text files (e.g., `Shopping.txt`)
- PDF documents
- Markdown files

5. **Run the application**
```bash
uv run python main.py
```

## 📖 Usage

### Interactive Mode

Once started, the application provides an interactive console where you can ask questions:

```
🔍 Sua pergunta: Who is Ash Ketchum?
```

### Available Commands

- **`sair`** or **`exit`**: Exit the application
- **`limpar`** or **`clear`**: Clear the console
- **`debug`**: Toggle debug mode
- **`memória`** or **`memory`**: View conversation history
- **`limpar_memória`** or **`clear_memory`**: Clear conversation history
- **Ctrl+C**: Interrupt execution

### Example Session

```
⚙️  INICIALIZANDO POKECONSULTOR
============================================================

[1] 📂 Carregando RAG service ...
[2] 🤖 Inicializando LLM (groq/llama-3.1-8b-instant)...
[3] 🎯 Configurando AI Agent com RAG...

✅ Sistema pronto para consultas!

============================================================
🎮 POKECONSULTOR - CONSULTOR DE POKÉMON COM IA
============================================================

💬 Faça suas perguntas sobre Pokémon!

🔍 Sua pergunta: Tell me about trainers from Cerulean City
```

## 🏗️ Architecture

### Project Structure

```
pokeconsultor/
├── agents/
│   └── ai_agent.py                    # LLM agent with memory
├── llm/
│   └── base.py                        # LLM profile management
├── models/
│   └── llm.py                         # Data models (LLMRequest, ConversationMessage)
├── services/
│   ├── logger.py                      # Logging configuration
│   ├── memory.py                      # Conversation memory management
│   ├── data_loaders/
│   │   ├── base.py                    # Abstract loader interface
│   │   ├── csv_loader.py              # CSV file loader
│   │   ├── pdf_loader.py              # PDF file loader
│   │   ├── text_loader.py             # Text/Markdown loader
│   │   └── factory.py                 # Loader factory pattern
│   └── rag/
│       ├── service.py                 # RAG service orchestration
│       ├── embeddings.py              # Embedding models management
│       ├── formatting/
│       │   ├── context.py             # Context formatting & chunking
│       │   └── tokenizer.py           # Token counting utilities
│       └── search/
│           ├── executor.py            # Search execution engine
│           ├── lexical.py             # Lexical/BM25 search
│           └── vector.py              # Vector/semantic search with FAISS
├── config.py                          # Application settings
main.py                               # Entry point
```

### Key Components

#### 1. RAG Service (`services/rag/`)
Modular Retrieval-Augmented Generation system with the following components:

**`service.py` - RAG Service Orchestration**
- Coordinates document loading and vector store management
- Executes search queries combining multiple strategies
- Manages FAISS vector stores with automatic cache invalidation
- Dynamically adjusts context based on LLM model capabilities

**`embeddings.py` - Embedding Models Management**
- Loads and manages embedding models
- Supports multiple embedding providers
- Caches embeddings for performance

**`formatting/` - Context Processing**
- `context.py`: Chunking strategies and context formatting
- `tokenizer.py`: Token counting utilities for context window management

**`search/` - Search Execution**
- `executor.py`: Coordinates different search strategies
- `lexical.py`: BM25/lexical search for exact matches
- `vector.py`: Semantic search using FAISS vector stores
- Hybrid search combining both approaches for better results

#### 2. AI Agent (`agents/ai_agent.py`)
- Wraps LangChain chat models
- Integrates conversation memory
- Handles multi-turn interactions
- Supports system messages and RAG context injection

#### 3. Data Loaders (`services/data_loaders/`)
- **Factory Pattern**: Automatic file type detection
- **CSV Loader**: Converts tabular data to searchable text
- **PDF Loader**: Extracts text from PDF documents
- **Text Loader**: Handles TXT and Markdown files
- **Extensible**: Easy to add new loader types

#### 4. LLM Profiles (`llm/base.py`)
- Manages multiple LLM configurations
- Supports different providers (Groq, OpenAI, etc.)
- Profile-based model selection (default, executor, supervisor)

#### 5. Conversation Memory (`services/memory.py`)
- Maintains conversation history
- Automatic history trimming
- Formatted output for LLM APIs

#### 6. Logging (`services/logger.py`)
- Centralized logging configuration
- Structured logging across all modules

## 🔧 Configuration

### LLM Providers

The system supports multiple LLM providers through LangChain:

- **Groq**: Fast inference with Llama models
- **OpenAI**: GPT models
- **HuggingFace**: Open-source models
- **Anthropic**: Claude models

Configure providers in your `.env` file with appropriate API keys.

### Model Context Windows

The RAG service automatically adjusts retrieval based on known model context windows:

| Model | Context Window | RAG Context (30%) |
|-------|----------------|-------------------|
| llama-3.1-8b-instant | 8,192 | ~2,457 tokens |
| llama-3.1-70b-versatile | 8,192 | ~2,457 tokens |
| mixtral-8x7b-32768 | 32,768 | ~9,830 tokens |
| gpt-4-turbo | 128,000 | ~38,400 tokens |

Unknown models default to 4,000 tokens for safety.

### Embedding Model

Default: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

This model provides excellent multilingual support. You can change it by modifying the `embedding_model` parameter in `RAGService`.

### Incremental Embedding

The system supports **intelligent incremental embedding** with file-level tracking using ChromaDB metadata.

#### How It Works

1. **ChromaDB Metadata**: On first run, all files are embedded and tracked via ChromaDB's internal metadata.
2. **Smart Detection**: On subsequent runs, the system:
   - Scans your data directory
   - Compares file hashes against those stored in ChromaDB metadata
   - Identifies: NEW files, MODIFIED files, DELETED files, UNCHANGED files
3. **Intelligent Strategy**:
   - **No changes**: Load from ChromaDB immediately (seconds)
   - **Minor changes** (< 30%): Skip embedding, use cached data
   - **Major changes** (≥ 30%): Rebuild entire index
   - Threshold: 30% (configurable in future if needed)

#### Example Workflow

```
First Run (cold start):
  1. Scan data/ directory → finds 10 files
  2. All files embedded and tracked in ChromaDB

Second Run (1 file changed):
  1. ChromaDB metadata → sees 10 files
  2. Check current files → 10 files exist
  3. Hash comparison → 1 file modified, 9 unchanged
  4. Change rate = 10% < 30% threshold → SKIP EMBEDDING
  5. Load from ChromaDB (instant) ✨

Third Run (5 new files added):
  1. ChromaDB metadata → sees 10 files
  2. Check current files → 15 files exist
  3. Status detection → 5 NEW files
  4. Change rate = 33% > 30% threshold → REBUILD
  5. Embed all 15 files and update ChromaDB
```

#### Key Benefits

- ⚡ **Speed**: Unchanged files load from ChromaDB in milliseconds
- 🎯 **Smart**: Automatically decides REBUILD vs SKIP based on change rate
- 💾 **Efficient**: Avoids unnecessary reprocessing
- 📊 **Auditable**: All decisions logged for transparency

#### Under the Hood

The implementation is elegantly simple:

- **EmbeddingService**: Loads, chunks, and embeds documents, tracking via ChromaDB metadata
- **RAGService**: Orchestrates the workflow - detects changes and decides strategy

This clean separation means:
- No invasive changes to embedding logic
- Easy to test and maintain
- Can extend with per-file embedding later if needed
- Fully backward compatible

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
uv run pytest tests/ -v

# Run fast tests (skip slow and integration tests)
uv run pytest tests/ -v -m "not slow and not integration"

# Run with coverage
uv run pytest tests/ -v --cov=pokeconsultor
```

Or use the configured tasks:

```bash
# Run tests
uv run task test

# Run fast tests
uv run task test-fast

# Type checking
uv run task type-check

# Format code
uv run task format
```

## 🎨 Customization

### Adding New Data Sources

1. Place your files in the `data/` directory
2. The system automatically detects and loads supported formats
3. Clear cache if needed: delete `.cache/vector_stores/`

### Creating Custom Loaders

To support new file formats, create a new loader class:

```python
from pokeconsultor.services.data_loaders.base import DataLoader

class MyCustomLoader(DataLoader):
    @staticmethod
    def supports(file_path: Path) -> bool:
        return file_path.suffix.lower() == '.myformat'
    
    def load(self, file_path: Path) -> list[str]:
        # Your loading logic here
        return documents
```

Then register it in `LoaderFactory`.

### Configuring LLM Profiles

Edit your `.env` file to add or modify LLM profiles. You can configure:
- Provider (groq, openai, anthropic, etc.)
- Model name
- Temperature (0.0-2.0)
- Max tokens
- Timeout settings

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes with appropriate tests
4. Run tests and type checking
5. Submit a pull request

## 📝 License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

### You are free to:

- ✅ **Share**: Copy and redistribute the material in any medium or format
- ✅ **Adapt**: Remix, transform, and build upon the material

### Under the following terms:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.

- **NonCommercial**: You may not use the material for commercial purposes.

- **No additional restrictions**: You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

For more details, see the [full license text](https://creativecommons.org/licenses/by-nc/4.0/legalcode).

### Citation

If you use this project, please cite:

```
PokeConsultor - AI-powered RAG Consultant System
Author: Fábio Radicchi Belotto
URL: https://github.com/frbelotto/PokeConsultor
Year: 2025
```

## 🙏 Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) for LLM orchestration
- Vector storage powered by [FAISS](https://github.com/facebookresearch/faiss)
- Embeddings from [Sentence Transformers](https://www.sbert.net/)
- Dependency management with [uv](https://github.com/astral-sh/uv)

## 📬 Contact

**Fábio Radicchi Belotto**
- Email: fabio_belotto@hotmail.com
- GitHub: [@frbelotto](https://github.com/frbelotto)

## 🗺️ Roadmap

- [ ] Web interface for easier interaction
- [ ] Support for audio/video transcription
- [ ] Multi-language UI support
- [ ] Export conversation history
- [ ] Advanced RAG strategies (hybrid search, re-ranking)
- [ ] Integration with more LLM providers
- [ ] Docker containerization
- [ ] Cloud deployment guides

---

**Made with ❤️ by Fábio Radicchi Belotto**
