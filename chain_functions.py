import dotenv
#from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
#from langchain.schema.runnable import RunnablePassthrough
import os
import shutil #for high level file operations

def load_documents(data_path):
    doc_loader = PyPDFDirectoryLoader(data_path)
    docs = doc_loader.load()
    print(f"Loaded {len(docs)} documents.")
    return docs

#chunking
def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 100,
        length_function = len,
        add_start_index = True #flag to add start index to each chunk
    )

    chunks = text_splitter.split_documents(data)
    print(f"Split {len(data)} documents into {len(chunks)} chunks.")

# Print example of page content and metadata for a chunk
    #document = chunks[0]
    #print(document.page_content)
    #print(document.metadata)

    return chunks  # Return the list of split text chunks

chroma_path = "chroma_data/"
def save_to_chroma(chunks):
    # Clear out the existing database directory if it exists
    if os.path.exists(chroma_path):
        shutil.rmtree(chroma_path)

    vector_db = Chroma.from_documents(
        chunks,
        OpenAIEmbeddings(),
        persist_directory = chroma_path
    )

    #persisting the database to disk
    vector_db.persist()
    print(f"Saved {len(chunks)} chunks to {chroma_path}.")

    return vector_db

