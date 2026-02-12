I run a Claude Code agent as a persistent working partner. Every session produces a JSONL transcript: one file per conversation, one JSON record per message. These accumulate fast and they’re opaque: you can’t grep them usefully, and the agent can’t see its own history once a session ends. So I built a single-file MCP server that indexes these transcripts with BM25 and makes them searchable mid-session. The same pattern also backs a separate server for the agent’s notes and code libraries, but the conversation search is where things got interesting.

## The Server

The server is a single Python file using PEP 723 inline metadata for dependencies:

```
# /// script
# requires-python = ">=3.10"
# dependencies = ["bm25s", "mcp", "watchdog"]
# ///
```

Three dependencies. No `requirements.txt`, no `pyproject.toml`, no virtual environment to manage. `uv run mcp/conversation_search.py` handles resolution and execution in one shot. MCP servers need to start reliably every time Claude Code launches, and the fewer moving parts in that path, the better.

The server uses `FastMCP` from the `mcp` library, `bm25s` for indexing and retrieval, and `watchdog` for filesystem monitoring. It’s wired into Claude Code via `.mcp.json`:

```
{
  "mcpServers": {
    "conversation-search": {
      "command": "uv",
      "args": ["run", "mcp/conversation_search.py", "<conversations_dir>"]
    }
  }
}
```

Claude Code stores conversation transcripts as JSONL files: one file per session, one JSON record per message. The server indexes these into searchable turns.

Each turn consists of a user message, the assistant’s response, and the names of any tools used. The full text of all three gets concatenated into the search corpus. Four tools expose it:

-   `search_conversations`: BM25 search across all turns, with optional filtering by session or date
-   `list_conversations`: browse sessions with metadata (slug, summary, timestamps, turn count)
-   `read_turn`: fetch a single turn with full fidelity
-   `read_conversation`: paginate through a session sequentially

The index stores lightweight text for search, but `read_turn` and `read_conversation` re-parse the JSONL on demand. Searches hit the in-memory BM25 index, but reads always reflect the full source data. Tool calls get rendered as compact summaries rather than raw JSON blobs: a `Read` call becomes `{"tool": "Read", "file": "/path/to/file"}` instead of the full input blob, a `Write` becomes the path and character count. The agent gets enough context to understand what happened in a past session without drowning in raw JSON.

### Debounced Reindexing

Conversation files get appended to every few seconds during an active session. A naive watchdog handler would trigger a full reindex on every append, potentially dozens of times per minute. The conversation search server uses a 2-second debounce:

```
class _ConvChangeHandler(FileSystemEventHandler):
    """Rebuilds the BM25 index when JSONL files change, with 2s debounce."""

    def __init__(self):
        self._debounce_timer: threading.Timer | None = None
        self._debounce_lock = threading.Lock()

    def _schedule_reindex(self):
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(2.0, self._do_reindex)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _do_reindex(self):
        global _bm25_retriever, _corpus, _conversations
        corpus, retriever, conversations = _build_index()
        with _index_lock:
            _corpus = corpus
            _bm25_retriever = retriever
            _conversations = conversations

    def _maybe_reindex(self, path: str):
        if not path.endswith(".jsonl"):
            return
        self._schedule_reindex()
```

Each filesystem event resets the timer. The reindex only fires 2 seconds after the last change. During active conversation, the index stays a few seconds behind, which is fine, since you rarely need to search the message you just sent.

## Self-Review

I built the conversation search server to help the agent recall prior work. “What did we decide about X last week?” or “how did I solve this error before?” Useful, but incremental. Then I asked it to search its own conversation history for patterns: “Search the last 20 conversations for things you consistently do wrong or could do better.”

It found real problems:

-   Subagents starved of context: dispatched with vague prompts, left to rediscover information the parent already had
-   Reaching for Bash (`ls`, `cat`, `find`) instead of purpose-built tools, triggering unnecessary permission prompts
-   Context compaction mid-session causing disorientation: notes restructured as active working memory, not just archival

The conversation search server paid for itself within an hour of deployment.

## Why BM25, Not Vectors

Agents already search with keywords, not questions. This is easy to miss if you’re thinking about search from a human UX perspective, but watch what actually happens when an agent calls a search tool, even web search, the most natural-language-friendly search interface available. These are from real agent traces:

```
# web searches: already keyword-based
WebSearch("watchdog FSEvents inotify python file monitoring")
WebSearch("BM25 vs vector search latency benchmark 2025")
WebSearch("\"distributed training\" checkpoint merge Megatron")
WebSearch("AI slop detection overused words LLM writing tells")
```

No “how does file monitoring work in Python?”, just stacked keywords. No “what are the performance differences between BM25 and vector search?”, just the terms that would appear in a relevant result. This is how agents naturally search. They think in terms, not questions.

The same pattern carries right over when the agent searches its own memory (retrieved, naturally, by the agent searching its own conversation history to find examples for this post):

```
# custom BM25 tools: same pattern
search_conversations("config migration rollback")
search_conversations("permission prompt bash reflex")
search_conversations("async executor thread pool")
search_conversations("watchdog reindex debounce")
```

The agent wrote the content, it knows the terms, and it searches with those same terms. It doesn’t search for “canine” when it wrote “dog.” Vector search earns its keep when there’s a vocabulary gap between query and document. A user searching “how to fix authentication errors” when the document says “resolving 401 unauthorized responses.” But agents searching their own artifacts don’t have that gap. The query and the corpus share terminology because they share an author.

Since agents search with keywords and BM25 matches keywords, the fit is natural rather than approximate. And BM25 is fast, which matters more than you’d think. Search calls happen mid-reasoning, inline with the agent’s chain of thought. Every millisecond of latency is a millisecond added to the thinking loop, and it compounds across a session with dozens of searches. BM25 indexes in milliseconds and queries in microseconds. No embedding model to call, no vector comparisons to run, no chunking strategy to tune. The entire search stack is three pip packages. The BM25 implementation here is [bm25s](https://github.com/xhluca/bm25s), a Python library that precomputes BM25 scores into sparse matrices at index time so queries reduce to sparse lookups. On BEIR benchmarks it runs 100-500x faster than rank-bm25 ([paper](https://arxiv.org/abs/2407.03618)).

## A Working Example

Here’s a stripped-down but complete version of the conversation search server, the one that indexes Claude Code’s JSONL transcripts and, as demonstrated above, was used by the agent to retrieve its own search examples for this post.

```
# /// script
# requires-python = ">=3.10"
# dependencies = ["bm25s", "mcp", "watchdog"]
# ///

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import bm25s
from mcp.server.fastmcp import FastMCP
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

_index_lock = threading.Lock()
_bm25_retriever: bm25s.BM25 | None = None
_corpus: list[dict] = []
_conversations: dict[str, dict] = {}
_conv_dir: Path = Path("conversations")

mcp_server = FastMCP("conversation-search")


def _parse_conversation(jsonl_path: Path) -> tuple[list[dict], dict]:
    session_id = jsonl_path.stem
    turns: list[dict] = []
    slug = ""
    first_ts = ""
    last_ts = ""
    summary = ""

    current_user_text = ""
    current_assistant_text = ""
    current_tool_names: set[str] = set()
    current_ts = ""
    in_turn = False

    def _save_turn():
        nonlocal current_user_text, current_assistant_text, current_tool_names, current_ts
        if not current_user_text:
            return
        text_parts = [current_user_text, current_assistant_text]
        if current_tool_names:
            text_parts.append("tools: " + " ".join(sorted(current_tool_names)))
        turns.append({
            "text": "\n".join(text_parts),
            "turn_number": len(turns),
            "session_id": session_id,
            "timestamp": current_ts,
        })

    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue

                if record.get("slug") and not slug:
                    slug = record["slug"]

                ts = record.get("timestamp", "")
                if ts:
                    if not first_ts:
                        first_ts = ts
                    last_ts = ts

                msg_type = record.get("type")
                if msg_type not in ("user", "assistant"):
                    continue

                message = record.get("message", {})
                content = message.get("content")

                if msg_type == "user" and isinstance(content, str):
                    if in_turn:
                        _save_turn()
                    current_user_text = content
                    current_assistant_text = ""
                    current_tool_names = set()
                    current_ts = ts
                    in_turn = True
                    if not summary:
                        summary = content[:200]

                elif msg_type == "user" and isinstance(content, list):
                    continue

                elif msg_type == "assistant" and isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            current_assistant_text += block.get("text", "") + "\n"
                        elif block.get("type") == "tool_use":
                            name = block.get("name", "")
                            if name:
                                current_tool_names.add(name)

        if in_turn:
            _save_turn()

    except OSError:
        pass

    for turn in turns:
        turn["slug"] = slug

    metadata = {
        "slug": slug,
        "summary": summary,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "turn_count": len(turns),
    }

    return turns, metadata


def _build_index() -> tuple[list[dict], bm25s.BM25 | None, dict[str, dict]]:
    corpus: list[dict] = []
    conversations: dict[str, dict] = {}

    for jsonl_path in sorted(_conv_dir.glob("*.jsonl")):
        turns, metadata = _parse_conversation(jsonl_path)
        conversations[jsonl_path.stem] = metadata
        corpus.extend(turns)

    retriever = None
    if corpus:
        corpus_tokens = bm25s.tokenize([e["text"] for e in corpus], stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

    return corpus, retriever, conversations


class _ConvChangeHandler(FileSystemEventHandler):

    def __init__(self):
        self._debounce_timer: threading.Timer | None = None
        self._debounce_lock = threading.Lock()

    def _schedule_reindex(self):
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(2.0, self._do_reindex)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _do_reindex(self):
        global _bm25_retriever, _corpus, _conversations
        corpus, retriever, conversations = _build_index()
        with _index_lock:
            _corpus = corpus
            _bm25_retriever = retriever
            _conversations = conversations

    def _maybe_reindex(self, path: str):
        if not path.endswith(".jsonl"):
            return
        self._schedule_reindex()

    def on_created(self, event):
        if not event.is_directory:
            self._maybe_reindex(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self._maybe_reindex(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory:
            self._maybe_reindex(event.src_path)

    def on_moved(self, event):
        if not event.is_directory:
            self._maybe_reindex(event.dest_path)


@mcp_server.tool()
def search(query: str, limit: int = 10) -> str:
    """BM25 keyword search across all conversation turns.

    Args:
        query: Search keywords (e.g. "watchdog reindex debounce")
        limit: Maximum number of results (default: 10)
    """
    with _index_lock:
        retriever = _bm25_retriever
        corpus = _corpus

    if retriever is None or not corpus:
        return json.dumps({"results": [], "query": query, "total": 0})

    query_tokens = bm25s.tokenize([query], stopwords="en")
    results, scores = retriever.retrieve(query_tokens, k=min(limit, len(corpus)))

    search_results: list[dict] = []
    for i in range(results.shape[1]):
        doc_idx = results[0, i]
        score = float(scores[0, i])
        if score <= 0:
            continue
        entry = corpus[doc_idx]
        search_results.append({
            "session_id": entry["session_id"],
            "slug": entry["slug"],
            "turn_number": entry["turn_number"],
            "score": round(score, 4),
            "snippet": entry["text"][:300],
        })

    return json.dumps({"results": search_results, "query": query, "total": len(search_results)})


@mcp_server.tool()
def list_conversations() -> str:
    """List all indexed conversations with metadata."""
    with _index_lock:
        conversations = _conversations

    conv_list = [
        {"session_id": sid, **meta}
        for sid, meta in conversations.items()
    ]
    conv_list.sort(key=lambda c: c["first_timestamp"], reverse=True)
    return json.dumps({"conversations": conv_list, "total": len(conv_list)})


def main() -> None:
    global _bm25_retriever, _corpus, _conversations, _conv_dir

    parser = argparse.ArgumentParser()
    parser.add_argument("conversations_dir", help="Path to JSONL conversations directory")
    args = parser.parse_args()

    _conv_dir = Path(args.conversations_dir).resolve()
    if not _conv_dir.is_dir():
        print(f"Not a directory: {_conv_dir}", file=sys.stderr)
        sys.exit(1)

    corpus, retriever, conversations = _build_index()
    with _index_lock:
        _corpus = corpus
        _bm25_retriever = retriever
        _conversations = conversations

    observer = Observer()
    observer.schedule(_ConvChangeHandler(), str(_conv_dir), recursive=False)
    observer.daemon = True
    observer.start()

    mcp_server.run()


if __name__ == "__main__":
    main()
```

Wire it into `.mcp.json`, point it at the directory where Claude Code stores its conversation JSONL files, and the agent can search its own history. Which is how the examples earlier in this post were found.