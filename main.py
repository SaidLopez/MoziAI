import gradio as gr
from groq import Groq
from dotenv import load_dotenv
from dbdump import initialize_chroma, split_text
import json

load_dotenv()  # Load environment variables from a .env file if present

# Initialize Everything
client = Groq()

with open('metadata.json', 'r', encoding='utf-8') as f:
        metadata = json.load(f)
clean_pages = split_text('100M Lost Chapters by Alex Hormozi.txt')
collection = initialize_chroma(clean_pages, metadata)  # Initialize ChromaDB collection


def format_history(history):    
    """Helper function to format Gradio's history for the LLM API."""    
    messages = []  
    for message in history:  
        # Handle new Gradio format: list of dicts with 'role' and 'content'  
        if isinstance(message, dict) and 'role' in message and 'content' in message:  
            messages.append({  
                "role": message['role'],   
                "content": message['content']  
            })  
        # Handle old Gradio format: tuples (user_msg, assistant_msg)  
        elif isinstance(message, (list, tuple)) and len(message) == 2:  
            user_msg, assistant_msg = message  
            messages.append({"role": "user", "content": user_msg})  
            if assistant_msg:  # Only add if assistant message exists  
                messages.append({"role": "assistant", "content": assistant_msg})  
    return messages

def chat_function(message, history):  
    """  
    This function is called when a user sends a message in the Gradio interface.  
    It performs the RAG steps: Retrieve, Augment, Generate.  
    """  
    
    if not history:  
        # If there's no history, the search query is the user's message itself  
        search_query = message  
    else:  
        # If there is history, we ask the LLM to create a standalone query  
        rewrite_prompt = (  
            "Given the following chat history and the latest user question, "  
            "formulate a standalone question that can be used to search a knowledge base for relevant information. "  
            "Only output the reformulated question.\n\n"  
            f"Chat History:\n{history}\n\n"  
            f"Latest User Question: {message}\n\n"  
            "Reformulated Question:"  
        )
        try:  
            # Use a non-streaming call to get the rewritten query  
            rewrite_response = client.chat.completions.create(  
                model="llama-3.1-8b-instant",  
                messages=[{"role": "user", "content": rewrite_prompt}],  
                temperature=0,  
            )  
            search_query = rewrite_response.choices[0].message.content.strip()  
        except Exception as e:  
            print(f"Error during query rewriting: {e}")  
            search_query = message # Fallback to the original message  
  
    print(f"Original query: '{message}'")  
    print(f"Rewritten search query: '{search_query}'")

        # Query ChromaDB for relevant context  
    results = collection.query(  
        query_texts=[search_query],  
        n_results=3  # Get the top 3 most relevant chunks  
    )  
    context_documents = results['documents'][0]  # Adjust based on how your client returns results
    context = "\n\n---\n\n".join(context_documents)  
  
    # 2. Augment  
    # Create the prompt for the LLM  
    system_prompt = (  
        "You are an AI assistant that embodies the business strategies and communication style of Alex Hormozi."  
        "Use the provided context from his book to answer the user's question. "  
        "Be direct, confident, and provide actionable advice. If the context doesn't contain the answer,"  
        "state that the specific information isn't in the provided material."  
        "Never mention the context or that you are receiving context from sources. Act as Alex Hormozi."
    )  

    # Format the history for the final API call  
    messages = [{"role": "system", "content": system_prompt}]  
    messages.extend(format_history(history))  
      
    # Add the final user prompt with context  
    user_prompt_with_context = (  
        f"Based on the following context, answer my question.\n\n"  
        f"Context:\n{context}\n\n"  
        f"Question: {message}"  
    )  
    messages.append({"role": "user", "content": user_prompt_with_context})
  
    # 3. Generate (with Streaming)  
    # We use a generator with `yield` to stream the response  
    try:  
        response_stream = client.chat.completions.create(  
            model="openai/gpt-oss-120b",  
            # model="llama-3.1-8b-instant",  
            messages=messages,  
            temperature=0.7,  
            stream=True,
        )  
          
        full_response = "" 
        for chunk in response_stream:
            if not chunk.choices:  
                continue
            delta = chunk.choices[0].delta  
            if delta and delta.content:  
                full_response += delta.content  
                yield full_response


    except Exception as e:  
        yield f"An error occurred: {e}"

    

demo = gr.ChatInterface(  
    fn=chat_function,  
    title="MoziAI",  
    description="I am the embodiment of Alex Hormozi's business acumen. Ask me anything about business growth, sales, marketing, or offers!",  
    type="messages"  
)  
  
if __name__ == "__main__":  
    demo.launch()  # Set reload=True for development to auto-reload on code changes
