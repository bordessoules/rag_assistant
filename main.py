import argparse
import logging
import os
from llm.task_manager import TaskManager
from loaders.factory import LoaderFactory
from vector_store.repository.chroma import ChromaRepository
from llm.lm_studio import LMStudioService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def should_process_file(file_path, supported_extensions):
    # Skip virtual environment directories
    if '.venv' in file_path.split(os.sep) or 'site-packages' in file_path.split(os.sep):
        return False
        
    # Skip pycache directories
    if '__pycache__' in file_path:
        return False
        
    # Check if file has supported extension
    return file_path.endswith(supported_extensions)

def format_source_document(doc):
    """Format a single source document with metadata and content"""
    content = doc.page_content
    preview = f"{content[:40]}\n...\n{content[-40:]}" if len(content) > 80 else content
    return f"""
Source: {doc.metadata.get('file_name', 'Unknown')}
Path: {doc.metadata.get('source', 'Unknown')}
{preview}
"""

def format_source_document_slim(doc):
    """Format a single source document with metadata and content"""
    return f"""Path: {doc.metadata.get('source', 'Unknown')}"""

def display_llm_response(response):
    """Display LLM response with proper Unicode handling"""
    print("\nLLM RESPONSE\n")
    try:
        print(response["answer"].encode('utf-8', errors='replace').decode('utf-8').strip())
    except UnicodeEncodeError:
        cleaned_response = ''.join(char for char in response["answer"] if ord(char) < 128)
        print(cleaned_response.strip())
        
    # Commented debug section for vector store analysis
    # if debug_mode:
    #     print("\nSOURCE DOCUMENTS")
    #     for idx, doc in enumerate(response["source_documents"], 1):
    #         print(f"\nDocument [{idx}]:{format_source_document_slim(doc)}")

def main():
    print("\n=== RAG Assistant Starting ===")
    parser = argparse.ArgumentParser(description='RAG Assistant CLI')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--website', type=str, help='URL to process and add to knowledge base')
    parser.add_argument('--websites-file', type=str, help='File containing list of URLs to process')
    parser.add_argument('--directory', type=str, help='Process all supported files in directory')
    parser.add_argument('--file-types', type=str, default='.py,.md,.json', 
                    dest='file_types',
                    help='Comma-separated file extensions to process (default: .py,.md,.json)')
    parser.add_argument('--max_depth', type=int, default=settings.WEB_MAX_DEPTH, 
                    help='Maximum depth for web crawling (default: %(default)s)'
)
    parser.add_argument('--query', type=str, help='Query to search')
    parser.add_argument('--reset', action='store_true', help='Reset the vector store')
    parser.add_argument('--prompt', type=str, help='Override default prompt template')
    parser.add_argument('--temperature', type=float, help='Set LLM temperature (0.0 to 1.0)')
    parser.add_argument('--max_tokens', type=int,
                    help='Maximum number of tokens to generate')
    parser.add_argument('--n_predict', type=int,
                    help='Maximum number of tokens to generate (legacy, same as max_tokens)')
    parser.add_argument('--search_k', type=int, 
                    default=settings.SEARCH_K,
                    help='Number of similar documents to retrieve')
    parser.add_argument('--chunk_size', type=int,
                    default=settings.CHUNK_SIZE,
                    help='Document chunk size')
    parser.add_argument('--chunk_overlap', type=int,
                    default=settings.CHUNK_OVERLAP,
                    help='Document chunk overlap')    
    parser.add_argument('--repeat_penalty', type=float,
                    help='Repetition penalty')
    parser.add_argument('--top_p', type=float,
                    help='Top-p sampling')
    parser.add_argument('--min_p', type=float,
                    help='Min-p sampling')
    parser.add_argument('--top_k', type=int,
                    help='Top-k sampling')
    parser.add_argument('--frequency_penalty', type=float,
                    help='Frequency penalty for token generation')
    parser.add_argument('--presence_penalty', type=float,
                    help='Presence penalty for token generation')
    parser.add_argument('--model_profile', type=str, 
                    default='default',
                    help='Name of the LLM profile to use')

    args = parser.parse_args()
    print(f"\nReceived arguments: {args}")
    logger.info(f"CLI Arguments: {args}")

    profile_name = args.model_profile if args.model_profile else args.model_name
    model_profile = settings.get_llm_profile(profile_name)

    generation_limit = args.max_tokens if args.max_tokens is not None else args.n_predict
    
    # Single parameter initialization block
    params = {
        'temperature': model_profile.temperature,
        'max_tokens': generation_limit if generation_limit is not None else model_profile.max_tokens,
        'top_p': model_profile.top_p,
        'min_p': model_profile.min_p,
        'top_k': model_profile.top_k,
        'repeat_penalty': model_profile.repeat_penalty,
    }

    # Override with CLI args if provided
    for key in params:
        if hasattr(args, key) and getattr(args, key) is not None:
            params[key] = getattr(args, key)

    # Initialize repository
    repository = ChromaRepository()
    # Handle reset first, before any initialization
    if args.reset:
        logger.debug("Starting reset operation")
        repository.reset()
        # Also reset file loader tracking
        file_loader = LoaderFactory.create('file')
        file_loader.reset_processed_files()
        logger.info("File loader tracking reset")

        # Reset web loader tracking
        web_loader = LoaderFactory.create('web')
        web_loader.reset_processed_files()
        logger.info("Vector store and all document tracking have been reset")
        return
    
    # Process single file
    if args.file:
        loader = LoaderFactory.create('file')
        documents = loader.load_file(args.file)
        if documents:
            repository.add_documents(documents)
            logger.info(f"Processed and stored file: {args.file}")

    # Process website
    if args.website:
        loader = LoaderFactory.create('web')
        documents = loader.process_website(
            args.website, 
            max_depth=args.max_depth
        )
        if documents:
            repository.add_documents(documents)
            logger.info(f"Processed and stored website: {args.website}")

    # Process websites from file
    if args.websites_file:
        with open(args.websites_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        loader = LoaderFactory.create('web', max_depth=args.max_depth)
        for url in urls:
            documents = loader.process_website(url)
            if documents:
                repository.add_documents(documents)
        logger.info(f"Processed {len(urls)} websites from file")
    
    # Process directory
    if args.directory:
        logger.info(f"Processing directory: {args.directory}")
        supported_extensions = tuple(args.file_types.split(','))
        loader = LoaderFactory.create('file')
        for root, _, files in os.walk(args.directory):
            for file in files:
                file_path = os.path.join(root, file)
                if should_process_file(file_path, supported_extensions):
                    logger.debug(f"Processing file: {file_path}")
                    documents = loader.load_file(file_path)
                    if documents:  # Only add if documents were successfully loaded
                        repository.add_documents(documents)
                        logger.info(f"Processed file: {file_path}")
        
    if args.query:
        logger.info(f"Processing query: {args.query}")
        logger.info(
            f"Using parameters:\n" + 
            "\n".join(f"- {k.title()}: {v}" for k, v in params.items())
        )
        # Get the retriever with search params
        retriever = repository.get_retriever(search_k=args.search_k)   
        llm_service = LMStudioService(
            vector_store=retriever, 
            **params
        )
        
        # Handle custom prompt
        if args.prompt:
            logger.info(f"Using custom prompt: {args.prompt}")
            llm_service.set_prompt_template(args.prompt)
        
        task_manager = TaskManager(llm_service)
        response = task_manager.process_query(args.query)
        display_llm_response(response)
if __name__ == "__main__": 
    main()