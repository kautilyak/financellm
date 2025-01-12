# src/main.py
import asyncio
from config import Config
from financial_agent.src.agents.agent_supervisor import AgentSupervisor
from rag.rag_system import RAGSystem
from data_providers.alpha_vantage import AlphaVantageProvider
from pprint import pprint


async def main():
    # Initialize system
    config = Config()
    agent_manager = AgentSupervisor()
    rag_system = RAGSystem(config)
    alpha_vantage = AlphaVantageProvider(config.alpha_vantage_key)

    # Load knowledge base
    rag_system.load_knowledge_base("./pythonProject/financial_agent/docs")

    # Example query with stock data
    while True:
        symbol = input("Enter a symbol for research: ")
        # Get market data
        stock_data = await alpha_vantage.get_stock_data(symbol)

        query = f'What\'s the current trend for {symbol} and what technical indicators should I consider?'
        rag_response = await rag_system.query(
            query,
            stock_data
        )

        # Get responses from agents
        agent_responses = await agent_manager.process_query(query, stock_data)

        # Print results
        pprint(f"Technical Analysis: {agent_responses['technical']['analysis']}")
        pprint(f"Additional Context: {rag_response}")

if __name__ == "__main__":
    asyncio.run(main())