# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..llm.ollama_integration import OllamaLLM


class BaseAgent(ABC):
    def __init__(self, name: str, description: str, model: str):
        """
        Initialize base agent with name and description.

        Args:
            name: Agent's identifier
            description: Agent's role and capabilities
        """
        self.name = name
        self.description = description
        self.llm = OllamaLLM(model_name=model)
        self.memory: List[Dict] = []  # Simple memory implementation

    def _add_to_memory(self, interaction: Dict[str, Any]):
        """Store interaction in agent's memory"""
        self.memory.append(interaction)
        # Keep only last 10 interactions to manage context window
        if len(self.memory) > 10:
            self.memory.pop(0)

    def _get_memory_context(self) -> str:
        """Get relevant context from memory"""
        return "\n".join(
            f"{m['role']}: {m['content']}"
            for m in self.memory[-3:]  # Last 3 interactions
        )

    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and generate response"""
        pass
