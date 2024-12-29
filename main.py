import argparse
import logging
import os
from loaders.base_loader import DocumentLoader
from vector_store.chroma_store import ChromaStore
from llm.lm_studio import LMStudioService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
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
    parser = argparse.ArgumentParser(description='RAG Assistant CLI')
    parser.add_argument('--file', type=str, help='Single file to process')
    parser.add_argument('--website', type=str, help='URL to process and add to knowledge base')
    parser.add_argument('--websites-file', type=str, help='File containing list of URLs to process')
    parser.add_argument('--directory', type=str, help='Process all supported files in directory')
    parser.add_argument('--file-types', type=str, default='.py,.md,.json', 
                    dest='file_types',
                    help='Comma-separated file extensions to process (default: .py,.md,.json)')
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
    logger.info(f"CLI Arguments: {args}")

    profile_name = args.model_profile if args.model_profile else args.model_name
    model_profile = settings.get_llm_profile(profile_name)

    generation_limit = args.max_tokens if args.max_tokens is not None else args.n_predict


    search_params = {'k': args.search_k if args.search_k is not None else settings.SEARCH_K}

    
    # Single parameter initialization block
    params = {
        'temperature': model_profile.temperature,
        'max_tokens': model_profile.max_tokens,
        'top_p': model_profile.top_p,
        'min_p': model_profile.min_p,
        'top_k': model_profile.top_k,
        'repeat_penalty': model_profile.repeat_penalty,
    }

    # Override with CLI args if provided
    for key in params:
        if hasattr(args, key) and getattr(args, key) is not None:
            params[key] = getattr(args, key)

    # Handle reset first, before any initialization
    if args.reset:
        store = ChromaStore()
        store.reset_store()
        DocumentLoader().reset_processed_files()
        logger.info("Vector store and document tracking have been reset")
        return
    
    store = ChromaStore()
    logger.info("Vector store initialized")
        
    if args.file:
        loader = DocumentLoader()
        documents = loader.load_file(args.file)
        store.add_documents(documents)
        logger.info(f"Processed and stored {len(documents)} chunks")
    if args.website:
        logger.info(f"Processing website: {args.website}")
        store = ChromaStore()
        store.add_website_content([args.website])
        
    if args.websites_file:
        with open(args.websites_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        logger.info(f"Processing {len(urls)} websites from file")
        store = ChromaStore()
        store.add_website_content(urls)
    if args.directory:
        supported_extensions = tuple(args.file_types.split(','))
        for root, _, files in os.walk(args.directory):
            for file in files:
                file_path = os.path.join(root, file)
                if should_process_file(file_path, supported_extensions):
                    logger.info(f"Processing file: {file_path}")
                    loader = DocumentLoader()
                    documents = loader.load_file(file_path)
                    if documents:  # Only add if documents were successfully loaded
                        store.add_documents(documents)
        
    if args.query:
        logger.info(f"Processing query: {args.query}")
        logger.info(
            f"Using parameters:\n" + 
            "\n".join(f"- {k.title()}: {v}" for k, v in params.items())
        )
        # Get the retriever with search params
        retriever = store.vector_store.as_retriever(search_kwargs={'k': args.search_k})
        llm_service = LMStudioService(
            vector_store=retriever, 
            **params
        )

        if args.prompt:
            logger.info(f"Using custom prompt: {args.prompt}")
            llm_service.set_prompt_template(args.prompt)
        
        response = llm_service.get_response(args.query)
        display_llm_response(response)  # Use the new display function
if __name__ == "__main__":
    main()