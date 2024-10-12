from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import pymongo
from dotenv import load_dotenv
import os
import argparse
load_dotenv('.env')

MONGO_URI = os.getenv("MONGO_URI") # Retrieve MongoDB URI from environment variables
HF_KEY = os.getenv("HF_KEY") # Hugging Face API key for embeddings model
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = pymongo.MongoClient(MONGO_URI)
DB_NAME = "RAG"
COLLECTION_NAME = "rag_pipeline"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "ragpipeline"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

models = [
    "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768"
]

def Query_Mongodb(query:str, model:str) ->str:

    # Initialize the LLM (ChatGroq) based on the provided model and temperature    
    llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model,
            temperature=0.1,
            max_tokens=1024
        )
 
    # Set up the embedding model using Hugging Face Inference API for text embedding
    embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=HF_KEY,
            model_name="BAAI/bge-small-en-v1.5"
        )
    
    # Initialize MongoDB-based vector store for retrieving document embeddings
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
            connection_string=MONGO_URI,
            namespace="RAG" + "." + "rag_pipeline",
            embedding=embeddings,
            index_name = ATLAS_VECTOR_SEARCH_INDEX_NAME
        )

     # Configure the retriever to retrieve the top 5 most similar documents based on embeddings  
    embeddings_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don’t justify your answers.
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    format_doc = (lambda docs: "\n\n".join([d.page_content for d in docs]))

    rag_chain = (
        {"context":embeddings_retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(query)

    return response


if __name__ == "__main__":

    query = input("Ask me anything: ")
    model = input("Enter model name: ")
    
    response = Query_Mongodb(query, model)
    print(response)