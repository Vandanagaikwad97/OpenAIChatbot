from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone
import time
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
from langchain_community.document_loaders import PyPDFLoader

def doc_preprocessing():
    current_dir = '/opt/render/project/src/data'
    pdf_path = os.path.join(current_dir, 'test_info.pdf')
    
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"Loaded {len(pages)} pages from PDF")
        
        for i, page in enumerate(pages):
            print(f"Page {i + 1} preview: {page.page_content[:100]}...")
        
        text_splitter = CharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        docs_split = text_splitter.split_documents(pages)
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
    
    index_name = 'marathichatbot'
    if index_name not in pinecone.list_indexes():
        print(f"Creating new index: {index_name}")
        pinecone.create_index(index_name, dimension=1536)
    else:
        print(f"Index {index_name} already exists")
    
    index = pinecone.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")
    
    if stats['total_vector_count'] == 0:
        docs_split = doc_preprocessing()
        print(f"Preprocessed {len(docs_split)} document chunks")
        
        if not docs_split:
            print("No documents to process. Returning None.")
            return None
        
        print("Starting to create Pinecone index")
        doc_db = Pinecone.from_documents(
            docs_split,
            embeddings,
            index_name=index_name,
        )
    else:
        print("Using existing Pinecone index")
        doc_db = Pinecone.from_existing_index(index_name, embeddings)
    
    print("Pinecone index ready")
    return doc_db
llm = ChatOpenAI()
doc_db = embedding_db()

def translate_to_marathi(text):
    translation_prompt = f"Translate the following English text to Marathi:\n\n{text}\n\nMarathi translation:"
    translation = llm.predict(translation_prompt)
    return translation


# Update retrieval_answer function to accept doc_db as an argument
def retrieval_answer(query, doc_db):
    try:
        retriever = doc_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(query)
        if docs:
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type='stuff',
                retriever=retriever,
            )
            result = qa.run(query)
        else:
            print("No relevant documents found. Falling back to general knowledge.")
            result = llm.predict(f"Please answer this question to the best of your ability: {query}")
        return result
    except Exception as e:
        print(f"Error in retrieval_answer: {str(e)}")
        return f"An error occurred: {str(e)}"


def main():
    st.title("Marathi Chatbot")
    
    start_time = time.time()
    doc_db = embedding_db()
    print(f"Time to initialize doc_db: {time.time() - start_time:.2f} seconds")
    
    if doc_db is None:
        st.error("Unable to initialize document database. Please check your PDF file and try again.")
        return
    
    text_input = st.text_input("तुमचा प्रश्न विचारा...")
    if st.button("प्रश्न विचारा"):
        if len(text_input) > 0:
            st.info("तुमचा प्रश्न: " + text_input)
            
            start_time = time.time()
            english_answer = retrieval_answer(text_input, doc_db)
            print(f"Time to retrieve answer: {time.time() - start_time:.2f} seconds")
            
            start_time = time.time()
            marathi_answer = translate_to_marathi(english_answer)
            print(f"Time to translate: {time.time() - start_time:.2f} seconds")
            
            st.success(marathi_answer)

if __name__ == "__main__":
    main()
