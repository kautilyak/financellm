# src/agents/agent_manager.py
from typing import Dict, List, Any
from .base_agent import BaseAgent
from .technical_analyst import TechnicalAnalystAgent


class AgentSupervisor:
    def __init__(self) -> None:
        """Initialize agent manager with available agents"""
        self.agents: Dict[str, BaseAgent] = {
            'technical': TechnicalAnalystAgent(),
            # Add more agents as we create them
        }

    async def process_query(
            self,
            query: str,
            data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process a query using appropriate agents.

        Args:
            query: User's question or request
            data: Relevant data for analysis
        """
        results = {}

        # For now, we'll use all agents
        # Later, we can add agent selection logic
        for agent_name, agent in self.agents.items():
            try:
                agent_input = {
                    'query': query,
                    **data  # Include all data
                }

                results[agent_name] = await agent.process(agent_input)

            except Exception as e:
                results[agent_name] = {
                    'error': f"Error in {agent_name}: {str(e)}"
                }

        return results