from __future__ import annotations
import logging
import httpx
from typing import List, Dict, Any

logger = logging.getLogger("techstore.rag")

# Re-create the error class so app/main.py doesn't throw a NameError
class LLMConfigError(Exception):
    pass

class LLMService:
    def __init__(self):
        # Ollama runs locally on port 11434 by default
        self.ollama_url = "http://localhost:11434/api/chat"
        self.model = "llama3.2"
        logger.info(f"LLM service initialized with local Ollama model '{self.model}'.")

    async def generate_response(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        history: List[Dict[str, str]]
    ) -> str:
        """Generates a grounded response using local Ollama model."""
        # Format the retrieved chunks into a clean text block
        context_str = "\n\n".join([
            f"--- Section: {chunk['section_title']} ---\n{chunk['text_preview']}"
            for chunk in retrieved_chunks
        ])

        # System prompt enforcing grounding
        system_prompt = (
            "You are an expert customer support agent for TechStore Co.\n"
            "Your core rule is STRICT GROUNDING: You must answer the user's question "
            "using ONLY the facts provided inside the <knowledge_base> tags below.\n"
            "If the answer cannot be found in the knowledge base, you must politely reply: "
            "'I am sorry, but I do not have that information in my database.'\n"
            "Do not make up any policies, prices, or details under any circumstances."
        )

        # Structure the messages list for Ollama's chat API
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Add current user query wrapped with the context
        user_content = (
            f"<knowledge_base>\n{context_str}\n</knowledge_base>\n\n"
            f"User Question: {query}"
        )
        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,  # Set to False for a clean complete string response
            "options": {
                "temperature": 0.0  # Keep it deterministic to stop guessing
            }
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(self.ollama_url, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"Ollama returned error status: {response.status_code}")
                    return "I'm having trouble accessing my local brain right now. Please try again."
                
                result = response.json()
                return result["message"]["content"]

        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return "Local AI connection error. Is Ollama running?"

    # Alias to protect both method naming variations
    async def generate_answer(self, query: str, retrieved_chunks: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
        return await self.generate_response(query, retrieved_chunks, history)
