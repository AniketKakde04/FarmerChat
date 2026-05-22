import os
import glob
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define folder paths
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

# We use a multilingual model so a Marathi query can match English documents!
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embedding_model():
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def build_vector_db():
    """Reads all TXT and CSV files from data/raw/ and stores them in ChromaDB."""
    documents = []
    
    # 1. Load all TXT files
    txt_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.txt"))
    for file_path in txt_files:
        print(f"Loading text file: {os.path.basename(file_path)}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())

    # 2. Load all CSV files
    csv_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.csv"))
    for file_path in csv_files:
        print(f"Loading CSV file: {os.path.basename(file_path)}")
        # CSVLoader treats each row as a separate document
        loader = CSVLoader(file_path, encoding='utf-8')
        documents.extend(loader.load())

    if not documents:
        print("No documents found in data/raw/. Please add your TXT and CSV files.")
        return None

    # 3. Chunk the documents
    # We split large documents into smaller pieces so the AI can read just the relevant parts
    print(f"Total raw documents loaded: {len(documents)}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )
    chunked_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(chunked_docs)} manageable chunks.")

    # 4. Create and persist the ChromaDB
    print("Generating embeddings and saving to ChromaDB... (This may take a minute)")
    embeddings = get_embedding_model()
    
    vector_store = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    print(f"✅ Success! Vector database built and saved to {CHROMA_DB_DIR}")
    return vector_store

def get_retriever():
    """Returns the retriever for our LangGraph nodes to use later."""
    embeddings = get_embedding_model()
    vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    # Return top 3 most relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 7})

if __name__ == "__main__":
    # When you run this script directly, it will build the database
    build_vector_db()