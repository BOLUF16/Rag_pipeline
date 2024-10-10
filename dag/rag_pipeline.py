from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch 
from langchain_core.prompts import ChatPromptTemplate
from pymongo.operations import SearchIndexModel
import pymongo
from dotenv import load_dotenv
from datetime import datetime, timedelta
from textwrap import dedent
import os


load_dotenv()
MONGO_URI = os.environ["MONGO_URI"] # Retrieve MongoDB URI from environment variables
HF_KEY = os.environ["HF_KEY"] # Hugging Face API key for embeddings model

PDF_DIR = "../../data"
PROCESSED_FILES_LOG = "../../log/processed_files.log"

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

with DAG(
        "Rag_Pipeline",
        default_args=default_args,
        schedule_interval=timedelta(minutes=5),
        catchup=False,
        description= "A Dag to ingest, split and embed new pdfs into Mongodb every 5 minutes"
)as dag:
    
    dag.doc_md = __doc__

    def check_for_new_documents(**kwargs):

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
    
    def process_pdfs(new_pdfs):
        if not new_pdfs:
            logging.info("No new pdfs to process")
            return []
        
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

        
        vector_store = MongoDBAtlasVectorSearch(
            collection=MONGODB_COLLECTION,
            embedding=embeddings,
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
            relevance_score_fn="cosine",
        )
        
        logging.info("data uploaded")
        with open(PROCESSED_FILES_LOG, "a") as f:
            for pdf in new_pdfs:
                f.write(pdf + "\n")
    

    def create_vector_search(**kwargs):
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
         
        MONGODB_COLLECTION.create_search_index(model=search_index_model)
    
    check_for_new_documents_task = PythonOperator(
        task_id = "check_for_new_document",
        python_callable=check_for_new_documents,
    )
    check_for_new_documents_task.doc_md = dedent(
        """\
        #### checking task
        this task checks for new documents.
        """
    )

    process_pdfs_task = PythonOperator(
        task_id = "process_pdf",
        python_callable=process_pdfs,
    )
    process_pdfs_task.doc_md = dedent(
        """\
        #### processing task
        this task split and embed new documents into mongodb
        """
    )

    create_vector_search_task = PythonOperator(
        task_id = "create_vector_search",
        python_callable=create_vector_search,
    )
    create_vector_search_task.doc_md = dedent(
        """\
        #### create vector search task
        this task creates vector search index
        """
    )

check_for_new_documents_task >> process_pdfs_task >> create_vector_search_task