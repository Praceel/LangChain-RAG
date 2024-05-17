import dotenv
from chain_functions import load_documents, split_text, save_to_chroma
from prompts import retrieval_prompt
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableSequence

dotenv.load_dotenv()
#loading the data
data_path = "data/"
data = load_documents(data_path)
if not data:
    print("No documents found. Please check the data path and ensure it contains PDF files.")
    exit()

chunks = split_text(data)
if not chunks:
    print("No chunks created. Please check the documents and ensure they are correctly loaded.")
    exit()

chroma_data = save_to_chroma(chunks)

data_retriever = chroma_data.as_retriever(k=10) #k is the number of chunks to retrieve
chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

output_parser = StrOutputParser()

retrieval_prompt_template = retrieval_prompt()

# Create the chain for the second step correctly
output_chain = RunnableSequence(
    retrieval_prompt_template,
    chat_model,
    output_parser
)

if __name__ == "__main__":
    question = "What is Uncanny valley?"

    # Step 1: Retrieve relevant context
    try:
        context_docs = data_retriever.get_relevant_documents(question)
        print(f"Retrieved {len(context_docs)} documents for context.")
        
        # Combine document contents into a single string
        context_str = " ".join([doc.page_content for doc in context_docs])
        print(f"Context String: {context_str[:500]}...")  # Print the first 500 characters of the context string
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        exit()

    # Step 2: Pass context and question to the chain
    input_data = {"context": context_str, "question": question}
    try:
        result = output_chain.invoke(input_data)
        print("Chain Invocation Result:", result)
    except Exception as e:
        print(f"Error in invoking chain: {e}")