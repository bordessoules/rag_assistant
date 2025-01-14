from pathlib import Path
import sqlite3
from typing import  List, Optional
from rich.markdown import Markdown
import re
import click
from rich.console import Console
from rich.panel import Panel
from langchain_openai import OpenAI

from langchain.schema import HumanMessage, SystemMessage

console = Console()

ASCII_LOGO = """
╔═══╗╔═══╗╔═══╗  ╔══╗─╔══╗╔════╗╔═══╗
║╔═╗║║╔═╗║║╔═╗║  ║╔╗║─║╔╗║║╔╗╔╗║║╔══╝
║╚═╝║║║─║║║║─╚╝  ║╚╝╚╗║║║║╚╝║║╚╝║╚══╗
║╔╗╔╝║╚═╝║║║╔═╗  ║╔═╗║║║║║──║║──║╔══╝
║║║╚╗║╔═╗║║╚╩═║  ║╚═╝║║╚╝║──║║──║╚══╗
╚╝╚═╝╚╝─╚╝╚═══╝  ╚═══╝╚══╝──╚╝──╚═══╝

╔═══════════════════ COMMANDS ══════════════════╗
║ > rag add <path>    : Add files to knowledge  ║
║ > rag ask "query"   : Query knowledge base    ║
║ > rag chat          : Interactive chat mode   ║
║                                              ║
║ Options:                                     ║
║ --model <name>      : Select LLM model       ║
║ --temp <0-1>        : Temperature            ║
║ --top-p <0-1>       : Top P sampling         ║
║ --min-p <0-1>       : Min P threshold        ║
║ --repeat <1+>       : Repetition penalty     ║
║ --freq <0-2>        : Frequency penalty      ║
║ --pres <0-2>        : Presence penalty       ║
║ --max-tokens <n>    : Max response length    ║
║ --streaming         : Enable streaming       ║
║                                              ║
║ Pro Tip: Use {{file:path}} in queries for    ║
║         direct file context                  ║
╚══════════════════════════════════════════════╝
"""

class DocumentStore:
    def __init__(self, db_path: str = "documents.db"):
        self.conn = sqlite3.connect(db_path)
        self.setup_db()
    
    def setup_db(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                filepath TEXT PRIMARY KEY,
                content TEXT
            )
        """)
        
    def add_document(self, content: str, filepath) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?)",
            (filepath, content)
        )
        self.conn.commit()
        
    def fetch_document(self, filename: str) -> List[tuple]:
        # Replace * with % for SQL LIKE syntax
        sql_pattern = filename.replace('*', '%')
        cursor = self.conn.execute(
            "SELECT filepath, content FROM documents WHERE filepath LIKE ?",
            (f"%{sql_pattern}",)
        )
        return cursor.fetchall()

class FileLoader:
    @staticmethod
    def load_file(filepath: str) -> str:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.35,
        max_tokens: Optional[int] = None
    ):
        # OpenAI class requires max_tokens to be an integer, not None
        llm_params = {
            "base_url": base_url,
            "api_key": "not-needed",
            "temperature": temperature,
            "streaming": False,
            "model": "local-model"
        }
        # Only add max_tokens if it's specified
        if max_tokens is not None:
            llm_params["max_tokens"] = max_tokens
            
        self.llm = OpenAI(**llm_params)
    
    def ask(self, query: str, context: str = "") -> str:
        system_message = "You are a helpful assistant, inside an geeky app, that answers questions based on the provided context. please provide response in clean markdown format"
        prompt = f"""Context: {context}\n\nQuestion: {query}"""
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=prompt)
        ]
        return self.llm.invoke(messages)
def extract_file_contexts(query: str, store: DocumentStore) -> tuple[str, str]:
    """Extract file-specific contexts and modify the query"""
    file_patterns = re.findall(r'\{\{file:(.*?)\}\}', query)
    context_parts = []
    
    for file_pattern in file_patterns:
        results = store.fetch_document(file_pattern)
        for filepath, content in results:
            context_parts.append(f"File: {filepath}\n{content}\n")
    
    clean_query = re.sub(r'\{\{file:(.*?)\}\}', '', query).strip()
    return "\n".join(context_parts), clean_query
# CLI commands stay the same but with type hints
@click.group()
def cli() -> None:
    """RAG Lite - Your Knowledge Assistant"""
    console.print(Panel(ASCII_LOGO, style="bold green"))

@cli.command()
@click.argument('path', type=click.Path(exists=True))
def add(path: str) -> None:
    """Add files or directories to knowledge base"""
    store = DocumentStore()
    loader = FileLoader()
    
    path_obj = Path(path)
    if path_obj.is_file():
        files = [path_obj]
    else:
        files = list(path_obj.rglob('*'))
    
    with console.status("[bold green]Adding files...") as status:
        for file in files:
            if file.is_file():
                content = loader.load_file(str(file))
                store.add_document(
                    content=content,
                    filepath=str(file)
                )
                console.print(f"Added: {file}")

@cli.command()
@click.argument('query')
@click.option('--temp', default=0.5, help='Temperature (0-1)')
@click.option('--max-tokens', default=1024*8, type=int, help='Max tokens in response')
def ask(query: str, temp: float, max_tokens: Optional[int]) -> None:
    """Query your knowledge base"""
    store = DocumentStore()
    client = LLMClient(temperature=temp, max_tokens=max_tokens)
    
    with console.status("[bold green]Thinking..."):
        file_context, clean_query = extract_file_contexts(query, store)
        response = client.ask(clean_query, file_context)
         # Render the response with rich markdown formatting
        md = Markdown(response, code_theme="monokai")
        console.print("\n[bold green]🤖 Response:[/bold green]")
        console.print(md)

@cli.command()
@click.option('--temp', default=0.5, help='Temperature (0-1)')
@click.option('--max-tokens', default=1024*8, type=int, help='Max tokens in response')
def chat(temp: float, max_tokens: Optional[int]) -> None:
    """Interactive chat mode"""
    store = DocumentStore()
    client = LLMClient(temperature=temp, max_tokens=max_tokens)
    messages_history = []
    
    console.print("[bold green]🤖 Chat mode activated (type 'exit' to quit)[/bold green]")
    
    while True:
        query = console.input("[bold blue]You:[/bold blue] ")
        if query.lower().strip() == 'exit':
            console.print("[bold green]👋 Goodbye![/bold green]")
            break
            
        with console.status("[bold green]Thinking..."):
            # Get file-specific context first
            file_context, clean_query = extract_file_contexts(query, store)
            
            # Build a structured conversation context
            conversation_context = "\n".join([
                "Previous conversation:",
                *messages_history,
                "---",
                "File context:",
                file_context,
                "---",
                f"Current query: {clean_query}"
            ])
            
            response = client.ask(clean_query, conversation_context)
            messages_history.append(f"User: {query}")
            messages_history.append(f"Assistant: {response}")
            
            # Keep context window manageable
            if len(messages_history) > 10:  # Keep last 5 exchanges
                messages_history = messages_history[-10:]
                
            md = Markdown(response, code_theme="monokai")
            console.print(Panel(md, title="🤖 Assistant", border_style="green"))

if __name__ == '__main__':
    cli()
