"""
VectorDataBasePipeline it is a class to use the df files directly thorw fastapi from directly from user.
it will store the embedded data in the postgrace database that the ai will use atomatically.
"""
import psycopg2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import logging


class VectorDataBasePipeline:
    """
    Docstring for VectorDataBasePipeline
    - Parameter:pdf_file
    """

    def __init__(self, pdf_file):

        try:
            loder = PyPDFLoader(pdf_file)
            self.docs = loder.load()

            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            self.chunks = self.splitter.split_documents(self.docs)
            logging.info(f"chuked data{len(self.chunks)}")

            self.embeddings_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as error:
            logging.error(error)
            raise RuntimeError(error)

    def connect_to_vectordb_and_store(self) -> List[str]:
        """
        Docstring for connect_to_vectordb_and_store

        :param self- just call this function after class call 
        :use_case-```VectorDataBasePipeline(pdf_file).connect_to_vectordb_and_store()```
        """
        try:
            self.conn = psycopg2.connect(
                dbname="vector",
                user="ssp001",
                password="admin",
                host="localhost",
                port=5433
            )
            self.curs = self.conn.cursor()
            for doc in self.chunks:
                text = doc.page_content
                embedding = self.embeddings_model.embed_query(text)

                self.curs.execute(
                    """
                    INSERT INTO documents (content, embedding)
                    VALUES (%s, %s::vector)
                    """,
                    (text, embedding)
                )
            self.conn.commit()
            self.curs.close()
            self.conn.close()
        except Exception as error:
            logging.error(error)
            raise RuntimeError(error)
