# src/knowledge_base/rag_system.py
from typing import Dict
from langchain_ollama.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor


class RAGSystem:
    def __init__(self, config):
        self.config = config
        self.vector_store = VectorStoreManager(config)
        self.document_processor = DocumentProcessor()
        self.llm = ChatOllama(
            model=config.llm_model,
            temperature=0.3  # Lower temperature for more focused responses
        )

        # Define our base prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a successful Technical Analyst. Use the 
            following pieces of context to answer the user with well though out analysis. If you don't know the 
            answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            """),
            ("human", "Hello, How are you?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{question}"),
            ("human", "Let's think about this step by step. Give me the technical analysis on {symbol} and analyze "
                      "any trends that you notice based on the historical price data in the context"),
            ("ai", "Here's what I think about {symbol}..")
        ])

    def load_knowledge_base(self, docs_directory: str):
        """Load and process documents into the vector store"""
        documents = self.document_processor.process_financial_documents(docs_directory)
        self.vector_store.add_documents(documents)

    async def query(self, question: str, stock_data: Dict = None) -> str:
        """
        Generate a response using RAG.
        Optionally incorporates real-time stock data.
        """
        # Retrieve relevant documents
        relevant_docs = self.vector_store.similarity_search(question)

        # Prepare context
        context_texts = [doc.page_content for doc in relevant_docs]
        if stock_data:
            context_texts.append(f"Current stock prices for {stock_data['metadata']['2. Symbol']}: {stock_data['prices'].to_string()}")
            context_texts.append(f"Current stock summary for {stock_data['metadata']['2. Symbol']}: {stock_data['summary']}")
            context_texts.append(f"metadata: {stock_data['metadata']}")

        context = "\n\n".join(context_texts)

        # Generate response
        prompt = self.prompt_template.format(
            context=context,
            question=question,
            symbol = stock_data['metadata']['2. Symbol'],
        )

        response = await self.llm.ainvoke(prompt)
        return response
