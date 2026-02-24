"""
Main file for running RAG-assistant.

This is the entry point to the application. It does:
1. Load configuration
2. Initialize all components (cache, vector DB, RAG)
3. Add sample documents (on first run)
4. Interactive chat loop with user
"""

import os
import time
from dotenv import load_dotenv
from embeddings import EmbeddingStore, get_sample_documents
from rag import RAGAssistant
from cache import ResponseCache


def initialize_system():
    """
    Initialize all RAG system components.
    
    Returns:
        Tuple (embedding_store, rag_assistant, cache)
    """
    print("=" * 70)
    print("[INIT] INITIALIZING RAG-ASSISTANT")
    print("=" * 70)
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://openai.api.proxyapi.ru/v1")
    if not api_key:
        print("[WARN] OPENAI_API_KEY not found in environment variables!")
        print("   Create .env file and add: OPENAI_API_KEY=your_key_here")
        print("   Or set environment variable in system.")
        print()
    
    # 1. Initialize cache for storing responses
    print("\n[1/3] Initializing cache...")
    cache = ResponseCache(cache_file="cache.json")
    
    # 2. Initialize FAISS vector store
    print("\n[2/3] Initializing vector store...")
    embedding_store = EmbeddingStore(
        collection_name="rag_documents",
        persist_directory="./faiss_db",
        embedding_model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url
    )
    
    # Check if we need to add sample documents
    if embedding_store.ntotal == 0:
        print("\n[DOC] Database is empty. Adding sample documents...")
        sample_docs = get_sample_documents()
        embedding_store.add_documents(sample_docs)
    else:
        print(f"[OK] Database already has {embedding_store.ntotal} documents")
    
    # 3. Initialize RAG assistant
    print("\n[3/3] Initializing RAG assistant...")
    rag_assistant = RAGAssistant(
        embedding_store=embedding_store,
        model="gpt-3.5-turbo",
        temperature=0.7,
        base_url=base_url
    )
    
    print("\n" + "=" * 70)
    print("[OK] SYSTEM READY")
    print("=" * 70)
    
    return embedding_store, rag_assistant, cache


def answer_question(query: str, rag_assistant: RAGAssistant, cache: ResponseCache) -> str:
    """
    Answer user question using cache and RAG.
    
    Logic:
    1. Check cache - if answer exists, return it
    2. If not, perform RAG (search + generation)
    3. Save new answer to cache
    4. Return answer
    
    Args:
        query: User question
        rag_assistant: RAG assistant instance
        cache: Cache instance
        
    Returns:
        Answer to question
    """
    print("\n" + "=" * 70)
    print(f"[QUESTION] {query}")
    print("=" * 70)
    
    # Start timing
    start_time = time.time()
    
    # Step 1: Check cache
    print("\n[Step 1] Checking cache...")
    cached_answer = cache.get(query)
    
    if cached_answer:
        # Answer found in cache - return it
        elapsed = time.time() - start_time
        print(f"\n[TIME] Response time: {elapsed:.3f} seconds (from cache)")
        print("\n[CACHE] Answer from cache:")
        print("-" * 70)
        print(cached_answer)
        print("-" * 70)
        return cached_answer
    
    # Step 2: Not in cache - perform RAG
    print("\n[Step 2] Performing RAG (search + generation)...")
    
    try:
        answer, search_results = rag_assistant.generate_response(
            query=query,
            top_k=3,
            verbose=True
        )
        
        # Step 3: Save answer to cache
        print("\n[Step 3] Saving answer to cache...")
        cache.set(query, answer)
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        print(f"\n[TIME] Total response time: {elapsed:.3f} seconds")
        
        # Print final answer
        print("\n[ANSWER]:")
        print("-" * 70)
        print(answer)
        print("-" * 70)
        
        return answer
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        print(f"\n[ERROR] {error_msg}")
        return error_msg


def interactive_mode(rag_assistant: RAGAssistant, cache: ResponseCache):
    """
    Interactive chat mode with assistant.
    
    User can ask questions in loop until exit command.
    """
    print("\n" + "=" * 70)
    print("[CHAT] INTERACTIVE MODE")
    print("=" * 70)
    print("\nYou can ask questions to the assistant.")
    print("To exit enter: exit, quit, or q")
    print("\nAvailable commands:")
    print("  - cache - show cache info")
    print("  - clear_cache - clear cache")
    print("  - stats - show system statistics")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("\n[YOU]: ").strip()
            
            # Check exit commands
            if user_input.lower() in ['exit', 'quit', 'q', '']:
                print("\n[BYE] Goodbye!")
                break
            
            # Handle special commands
            if user_input.lower() == 'cache':
                print(f"\n[STATS] Cache contains {cache.size()} entries")
                continue
            
            if user_input.lower() == 'clear_cache':
                cache.clear()
                print("\n[OK] Cache cleared")
                continue
            
            if user_input.lower() == 'stats':
                print(f"\n[STATS] SYSTEM STATISTICS:")
                print(f"  - Documents in FAISS: {rag_assistant.embedding_store.ntotal}")
                print(f"  - Cache entries: {cache.size()}")
                print(f"  - LLM model: {rag_assistant.model}")
                continue
            
            # Handle user question
            answer_question(user_input, rag_assistant, cache)
            
        except KeyboardInterrupt:
            print("\n\n[BYE] Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {str(e)}")


def demo_mode(rag_assistant: RAGAssistant, cache: ResponseCache):
    """
    Demo mode with predefined questions.
    
    Shows system working on examples, including cache usage.
    """
    print("\n" + "=" * 70)
    print("[DEMO] DEMONSTRATION MODE")
    print("=" * 70)
    print("\nNow RAG-assistant will be demonstrated")
    print("on several example questions.\n")
    
    # Demo questions
    demo_questions = [
        "What is Python and what is it used for?",
        "Tell me about RAG and how it works",
        "What are vector databases?",
        "What is Python and what is it used for?"  # Repeat for cache demo
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n\n{'#' * 70}")
        print(f"QUESTION {i} of {len(demo_questions)}")
        print(f"{'#' * 70}")
        
        answer_question(question, rag_assistant, cache)
        
        # Pause between questions (except last)
        if i < len(demo_questions):
            input("\n[Press Enter for next question...]")
    
    print("\n\n" + "=" * 70)
    print("[OK] DEMONSTRATION COMPLETE")
    print("=" * 70)


def main():
    """
    Main application function.
    """
    try:
        # Initialize system
        embedding_store, rag_assistant, cache = initialize_system()
        
        # Mode selection
        print("\n" + "=" * 70)
        print("MODE SELECTION")
        print("=" * 70)
        print("\n1. Interactive mode - ask your own questions")
        print("2. Demo mode - ready-made example questions")
        print()
        
        mode = input("Select mode (1 or 2, default 1): ").strip()
        
        if mode == '2':
            demo_mode(rag_assistant, cache)
            
            # Offer to switch to interactive mode
            print("\n" + "=" * 70)
            continue_interactive = input("\nSwitch to interactive mode? (y/n): ").strip().lower()
            if continue_interactive in ['y', 'yes', '']:
                interactive_mode(rag_assistant, cache)
        else:
            interactive_mode(rag_assistant, cache)
        
    except Exception as e:
        print(f"\n[ERROR] Critical error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
