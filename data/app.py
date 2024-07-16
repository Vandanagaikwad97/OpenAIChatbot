from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

def doc_preprocessing():
    current_dir = '/opt/render/project/src/data'
    print(f"Current directory: {current_dir}")
    
    # List all files in the directory
    files = os.listdir(current_dir)
    print(f"Files in directory: {files}")
    
    # Check for PDF files
    pdf_files = [f for f in files if f.endswith('.pdf')]
    print(f"PDF files found: {pdf_files}")
    
    if not pdf_files:
        print("No PDF files found in the directory!")
        return []
    
    try:
        loader = DirectoryLoader(
            current_dir,
            glob='*.pdf',
            show_progress=True
        )
        print("DirectoryLoader initialized")
        
        docs = loader.load()
        print(f"Documents loaded: {len(docs)}")
        
        if not docs:
            print("No documents were loaded by DirectoryLoader!")
            return []
        
        # Print the first few characters of each loaded document
        for i, doc in enumerate(docs):
            print(f"Document {i + 1} preview: {doc.page_content[:100]}...")
        
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0
        )
        docs_split = text_splitter.split_documents(docs)
        print(f"Documents split into {len(docs_split)} chunks")
        
        return docs_split
    except Exception as e:
        print(f"Error in doc_preprocessing: {str(e)}")
        return []

# @st.cache_resource
def embedding_db():
    print("Starting embedding_db function")
    embeddings = OpenAIEmbeddings()
    print("OpenAI Embeddings initialized")
    
    pinecone.Pinecone(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )
    print("Pinecone initialized")
    
    docs_split = doc_preprocessing()
    print(f"Preprocessed {len(docs_split)} document chunks")
    
    if not docs_split:
        print("No documents to process. Returning None.")
        return None
    
    print("Starting to create Pinecone index")
    doc_db = Pinecone.from_documents(
        docs_split,
        embeddings,
        index_name='marathichatbot',
    )
    print("Pinecone index created successfully")
    return doc_db

llm = ChatOpenAI()
doc_db = embedding_db()

def translate_to_marathi(text):
    translation_prompt = f"Translate the following English text to Marathi:\n\n{text}\n\nMarathi translation:"
    translation = llm.predict(translation_prompt)
    return translation

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=doc_db.as_retriever(),
    )
    result = qa.run(query)
    marathi_result = translate_to_marathi(result)
    return marathi_result

def main():
    st.title("Marathi Chatbot")

    text_input = st.text_input("तुमचा प्रश्न विचारा...")

    if st.button("प्रश्न विचारा"):
        if len(text_input) > 0:
            st.info("तुमचा प्रश्न: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
