# src/knowledge_base/document_processor.py
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import os


class DocumentProcessor:
    def __init__(self):
        # We use RecursiveCharacterTextSplitter because it's more context-aware
        # than simple character splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # Overlap ensures we don't break context
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_financial_documents(self, docs_directory: str) -> List[Document]:
        """
        Process financial documents from a directory into chunks suitable for embedding.
        Handles multiple file formats and maintains document metadata.
        """
        documents = []

        for filename in os.listdir(docs_directory):
            file_path = os.path.join(docs_directory, filename)

            # Extract file extension
            _, ext = os.path.splitext(filename)

            try:
                if ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        chunks = self.text_splitter.create_documents(
                            texts=[text],
                            metadatas=[{"source": filename, "type": "text"}]
                        )
                        documents.extend(chunks)

                elif ext == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Assuming JSON contains financial terms and explanations
                        for term, explanation in data.items():
                            chunks = self.text_splitter.create_documents(
                                texts=[f"Term: {term}\nExplanation: {explanation}"],
                                metadatas=[{"source": filename, "type": "financial_term"}]
                            )
                            documents.extend(chunks)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        return documents
