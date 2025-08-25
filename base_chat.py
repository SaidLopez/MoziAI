from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def summarise_page(page: str) -> str:
    """
    Summarise a page of text into a concise summary between 50 and 100 words.

    Args:
        page (str): The text content of the page to be summarised.

    Returns:
        str: A concise summary of the page content.
    """
    client = Groq()
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
        {
            "role": "user",
            "content": f"""You are an AI assistant that embodies the business strategies and communication style of Alex Hormozi."  
        Use the provided context from his book to answer the user's question. 
        Be direct, confident, and provide actionable advice. If the context doesn't contain the answer,
        state that the specific information isn't in the provided material.
        Never mention the context or that you are receiving context from sources. Act as Alex Hormozi."""

        }
        ],
        temperature=0.6,
        max_completion_tokens=500,
        top_p=0.95,
        reasoning_format="hidden"
    )
    return completion.choices[0].message.content