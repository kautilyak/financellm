# src/llm/ollama_integration.py
from langchain_ollama.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, Dict


class OllamaLLM:
    def __init__(self, model_name: str = "llama2:3.2"):
        """
        Initialize Ollama LLM with specified model.

        Args:
            model_name: The name of the Ollama model to use
        """
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.7,
            # Customize token limits based on your needs
            context_window=4096,
            timeout=120
        )

        self.output_parser = StrOutputParser()

    async def generate_response(
            self,
            prompt: str,
            system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response using the Ollama model.

        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to set context
        """
        try:
            # Create a chat prompt template
            template = ChatPromptTemplate.from_messages([
                ("system", system_prompt or "You are a helpful AI assistant"),
                ("human", prompt)
            ])

            # Create and execute the chain
            chain = template | self.llm | self.output_parser
            response = await chain.ainvoke({"prompt": prompt})

            return response

        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")