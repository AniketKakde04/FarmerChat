from app.graph.state import GraphState
from app.database import get_retriever
from app.services.sarvam_api import generate_agri_response

def retrieve_node(state: GraphState):
    """Worker 1: Searches ChromaDB for relevant agricultural info."""
    print("--- [NODE] Retrieving documents from Vector DB ---")
    
    query = state["user_query"]
    retriever = get_retriever()
    
    # Fetch the top 3 most relevant chunks of text from our database
    docs = retriever.invoke(query)
    
    # Extract just the text content from the Document objects
    doc_texts = [doc.page_content for doc in docs]
    
    # Write the found documents onto our State notepad
    return {"context": doc_texts}

def generate_node(state: GraphState):
    """Worker 2: Sends the query and the documents to Sarvam AI."""
    print("--- [NODE] Generating Marathi response with Sarvam AI ---")
    
    query = state["user_query"]
    # Join our list of documents into one big string of text
    context_string = "\n\n".join(state["context"])
    
    # Call our Sarvam API, passing both the farmer's question AND the database text
    answer = generate_agri_response(user_message=query, context_data=context_string)
    
    # Write the final answer onto our State notepad
    return {"final_answer": answer}

def chat_node(state: GraphState):
    """Worker 3: Handles simple greetings without searching the database."""
    print("--- [NODE] Simple Greeting Detected. Skipping Database. ---")
    
    query = state["user_query"]
    
    # Call Sarvam API WITHOUT the context_data
    answer = generate_agri_response(user_message=query, context_data=None)
    
    return {"final_answer": answer}