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
        model="qwen/qwen3-32b",
        messages=[
        {
            "role": "user",
            "content": f"You are RAG engineer summarising chunks of text. Summarise the following text into a concise summary between 50 and 100 words:\n{page}"
        }
        ],
        temperature=0.6,
        max_completion_tokens=500,
        top_p=0.95,
        reasoning_format="hidden"
    )
    return completion.choices[0].message.content
