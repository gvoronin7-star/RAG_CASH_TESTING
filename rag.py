"""
Module for RAG (Retrieval-Augmented Generation) implementation.
"""

from typing import List, Tuple, Optional
from openai import OpenAI
import os


class RAGAssistant:
    """RAG assistant using vector search and LLM."""
    
    def __init__(
        self, 
        embedding_store,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        base_url: str = "https://openai.api.proxyapi.ru/v1"
    ):
        self.embedding_store = embedding_store
        self.model = model
        self.temperature = temperature
        
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url
        )
        
        print(f"[OK] RAG assistant initialized (model: {model})")
    
    def _format_context(self, search_results: List[Tuple[str, str, float]]) -> str:
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, (chunk_text, source, distance) in enumerate(search_results, 1):
            context_parts.append(f"[Document {i} - {source}]\n{chunk_text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        prompt = f"""You are a helpful AI assistant. Use the following information 
from the knowledge base to answer the user's question.

IMPORTANT: 
- Answer based on the provided context
- If context doesn't have info to answer, say so honestly
- Answer in Russian
- Be specific and informative

=== CONTEXT FROM KNOWLEDGE BASE ===
{context}

=== USER QUESTION ===
{query}

=== ANSWER ===
"""
        return prompt
    
    def generate_response(
        self, 
        query: str, 
        top_k: int = 3,
        verbose: bool = True
    ) -> Tuple[str, List[Tuple[str, str, float]]]:
        if verbose:
            print(f"\n[SEARCH] Searching relevant documents (top_k={top_k})...")
        
        search_results = self.embedding_store.search(query, top_k=top_k)
        
        if verbose and search_results:
            print(f"\n[DOCS] Found {len(search_results)} relevant fragments:")
            for i, (chunk, source, distance) in enumerate(search_results, 1):
                print(f"  {i}. [{source}] (similarity: {1 - distance:.3f})")
                print(f"     {chunk[:100]}...")
        
        context = self._format_context(search_results)
        prompt = self._create_prompt(query, context)
        
        if verbose:
            print(f"\n[LLM] Generating answer with {self.model}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content.strip()
            return answer, search_results
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            print(f"[ERROR] {error_message}")
            return error_message, search_results
    
    def simple_response(self, query: str) -> str:
        answer, _ = self.generate_response(query, verbose=False)
        return answer
