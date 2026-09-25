# Codebase Q&A Agent

A command-line agent that lets you **ask natural-language questions about a Python codebase**. It indexes every `.py` file in a target repo into a semantic vector store, then uses a tool-calling LLM agent (powered by [Groq](https://groq.com/)) to search that index and answer questions with references to specific files and functions.

## Features

- 🔍 **Semantic search over source code** — chunks your codebase with a Python-aware splitter and embeds it with a local HuggingFace model (`all-MiniLM-L6-v2`).
- 🤖 **Tool-calling agent** — the LLM decides when to search the codebase via a `search_codebase` retriever tool, rather than blindly stuffing context into every prompt.
- ⚡ **Fast inference** — uses Groq's `llama-3.3-70b-versatile` for low-latency responses.
- 💬 **Interactive REPL** — ask follow-up questions in a simple terminal chat loop.
- 📁 **Point it at any repo** — defaults to a local `sample_proj` folder, or pass any path with `--repo`.

## How it works

1. **Load** — recursively walks the target repo and reads every `.py` file into a `Document`.
2. **Chunk** — splits each file into ~256-character chunks using a Python-syntax-aware `RecursiveCharacterTextSplitter`.
3. **Embed & store** — embeds chunks with `HuggingFaceEmbeddings` and stores them in an in-memory [Chroma](https://www.trychroma.com/) vector store.
4. **Agent** — wraps the vector store as a retriever tool and hands it to a LangChain agent (`create_agent`) backed by Groq's LLM. The agent searches the codebase before answering and cites specific files/functions.
5. **Chat** — streams the agent's responses in a terminal loop until you type `exit` or `quit`.

## Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-name>
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### `requirements.txt`

```
python-dotenv
langchain
langchain-core
langchain-text-splitters
langchain-chroma
langchain-huggingface
langchain-groq
```

## Setup

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

## Usage

Run against the default sample project:

```bash
python codebase_agent.py
```

Or point it at any repository on your machine:

```bash
python codebase_agent.py --repo /path/to/your/project
```

You'll see something like:

```
Loaded 42 files - 318 chunks (chunk_size=256)
Ready . ask any question , type 'exit' to quit

You: what does the load_codebase function do?
Agent: The load_codebase function in codebase_agent.py recursively walks the
given repo path for .py files and loads each one into a Document with its
file path stored as metadata.
```

Type `exit` or `quit` to end the session.

## Project structure

```
.
├── codebase_agent.py   # main script
├── sample_proj/        # default repo used when --repo is not passed
├── .env                # your GROQ_API_KEY (not committed)
├── requirements.txt
└── README.md
```

## Configuration

| Constant / Arg | Description | Default |
|---|---|---|
| `CHUNK_SIZE` | Character size of each code chunk | `256` |
| `--repo` | Path to the codebase to index | `../sample_proj` |
| `k` (retriever) | Number of chunks retrieved per query | `4` |
| `model_name` (LLM) | Groq model used | `llama-3.3-70b-versatile` |

## Notes

- The vector store is **in-memory** — it's rebuilt every time you run the script. For large repos, consider persisting Chroma to disk.
- `.env` should be excluded from version control (add it to `.gitignore`).

## License

MIT — feel free to use and adapt.
