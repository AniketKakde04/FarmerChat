from langgraph.graph import StateGraph, START, END
from app.graph.state import GraphState
from app.graph.nodes.rag import retrieve_node, generate_node, chat_node

def route_query(state: GraphState):
    """The Decision Maker: Looks at the query to decide the path."""
    query = state["user_query"].lower()
    
    # List of common greetings (English and Marathi)
    greetings = ["hi", "hello", "hey", "namaste", "नमस्कार", "हाय", "हॅलो"]
    
    # If the message is short (under 4 words) and contains a greeting, just chat
    if len(query.split()) < 4 and any(g in query for g in greetings):
        print("--- [ROUTER] Route to: CHAT ---")
        return "chat_path"
    else:
        print("--- [ROUTER] Route to: DATABASE SEARCH ---")
        return "rag_path"

# 1. Create the graph and tell it to use our GraphState notepad
workflow = StateGraph(GraphState)

# 2. Add our workers (nodes) to the graph
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("chat", chat_node)

# 3. Draw the arrows connecting them
# Instead of a straight line from START, we use our router!
workflow.add_conditional_edges(
    START,
    route_query,
    {
        "chat_path": "chat",       # If router says 'chat_path', go to chat_node
        "rag_path": "retrieve"     # If router says 'rag_path', go to retrieve_node
    }
)

# 4. Finish the rest of the arrows
workflow.add_edge("retrieve", "generate") # After retrieval, pass data to Sarvam AI
workflow.add_edge("generate", END)        # After generation, we are done!
workflow.add_edge("chat", END)            # After a simple chat, we are also done!

# 5. Compile it into an application we can run
app_graph = workflow.compile()