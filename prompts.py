import dotenv
from langchain_openai import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
import os
import shutil #for high level file operations

dotenv.load_dotenv()

def retrieval_prompt():
    template_str = """ Your job is to clearly answer the questions asked by the user about psychology.
                                Use the following context to answer the questions.
                                Be as detailed as possible, but do not make up any information outside of the context.
                                If you don't know an answer, just say you don't know, and ask for more information.
                                """

    system_prompt = SystemMessagePromptTemplate(
            prompt = PromptTemplate(
                input_variables = ["context"],
                template = template_str,
            )
    )

    human_prompt = HumanMessagePromptTemplate(
            prompt = PromptTemplate(
                input_variables = ["question"],
                template = "{question}",
            )
    )

    messages = [system_prompt, human_prompt]

    retrieval_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages = messages,
    )

    return retrieval_prompt_template

#output_parser = StrOutputParser()
#retrieval_prompt_template = retrieval_prompt()
#chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

#retrieval_chain = retrieval_prompt_template | chat_model | output_parser


