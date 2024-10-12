from airflow.decorators import dag, task
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch 
from pymongo.operations import SearchIndexModel
import pymongo
from dotenv import load_dotenv
from datetime import datetime, timedelta
import os


load_dotenv('.env')
MONGO_URI = os.getenv("MONGO_URI") # Retrieve MongoDB URI from environment variables
HF_KEY = os.getenv("HF_KEY") # Hugging Face API key for embeddings model

PDF_DIR = "/usr/local/airflow/data"
PROCESSED_FILES_LOG = "/usr/local/airflow/log/processed_files.log"

client = pymongo.MongoClient(MONGO_URI)

DB_NAME = "RAG"
COLLECTION_NAME = "rag_pipeline"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "ragpipeline"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 10),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    default_args=default_args,
    schedule_interval=timedelta(minutes=5),
    catchup=False,
    description="A DAG to ingest, split, and embed new PDFs into MongoDB every 5 minutes",
)

def pdf_ingestion_and_embedding():

    @task
    def check_for_new_documents():

        if os.path.exists(PROCESSED_FILES_LOG):
            with open(PROCESSED_FILES_LOG, "r") as f:
                processed_files = f.read().splitlines()
        
        else:
            processed_files = []
        
        new_pdfs = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf") and f not in processed_files]

        if new_pdfs:
            logging.info(f"New pdfs found: {new_pdfs}")
            return new_pdfs
        else:
            logging.info("No new pdfs found")
            return []
    
    
    @task
    def process_pdfs(new_pdfs):
        if not new_pdfs:
            logging.info("No new pdfs to process")
            return 
        
        all_chunks = []

        for pdf in new_pdfs:
            filepath = os.path.join(PDF_DIR, pdf)
            loader = PyPDFLoader(filepath)
            pages = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 50)
            chunks = text_splitter.split_documents(pages)

            all_chunks.extend(chunks)
        
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_KEY,
            model_name="BAAI/bge-small-en-v1.5"
        )

        existing_indexes = MONGODB_COLLECTION.list_search_indexes()
        list_index = [f for f in existing_indexes]
           
        search_index_model = SearchIndexModel(
                    definition={
                        "fields": [
                            {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 384,
                            "similarity": "cosine"
                            },
                            {
                            "type": "filter",
                            "path": "page"
                            }
                        ]
                    },
                    name="ragpipeline",
                    type="vectorSearch"
                    )
                    
        logging.info("Generating embeddings")
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=all_chunks,
            collection=MONGODB_COLLECTION,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )
        if list_index == []:
            logging.info(f"Creating {ATLAS_VECTOR_SEARCH_INDEX_NAME} search index")
            MONGODB_COLLECTION.create_search_index(model=search_index_model)
            logging.info(f"{ATLAS_VECTOR_SEARCH_INDEX_NAME} search index created")
    
        logging.info("Embedding generation completed")

        logging.info("data uploaded")
        with open(PROCESSED_FILES_LOG, "a") as f:
            for pdf in new_pdfs:
                f.write(pdf + "\n")
       
    new_pdfs = check_for_new_documents()
    process_pdfs(new_pdfs)

dag = pdf_ingestion_and_embedding()