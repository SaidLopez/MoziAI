import chromadb
import re
from agents.summarise_chunks_agent import summarise_page

def initialize_chroma(data, metadata):

    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection("my_collection")
    print("ChromaDB initialized and collection created.")
    for i, (page, meta) in enumerate(zip(data, metadata)):
        collection.add(
            documents=[page],
            metadatas=[meta],
            ids=[f"page_{i+1}"]
        )
    print("Data added to ChromaDB collection.")
    return collection

def split_text(book):
    with open(book, 'r', encoding='utf-8') as file:
        book_text = file.read()
    print(f"Total characters in book: {len(book_text)}")
    copyright_base = r"Copyright\s+(?:©|\(c\))\s+\d{4}\s+by\s+BUMBLE\s+IP,\s+LLC\s+NOT\s+FOR\s+DISTRIBUTION"  
  
    # Create the two patterns we need to find.  
    # Pattern A: Number is at the beginning.  
    pattern_A = r"\d+\s+" + copyright_base  
    # Pattern B: Number is at the end.  
    pattern_B = copyright_base + r"\s+\d+"  
    
    # Combine them with the '|' (OR) operator.  
    # This tells the regex engine to match EITHER pattern_A OR pattern_B.  
    # We wrap them in parentheses to group them.  
    split_pattern = f"(?:{pattern_A})|(?:{pattern_B})"  #Add non-capturing groups with (?:...) to avoid including the delimiters in the results.
    
    # Use re.split() to break the text into a list based on the combined pattern  
    pages = re.split(split_pattern, book_text)
    cleaned_pages = [page.strip() for page in pages if page and page.strip()]

    print(f"Total pages found: {len(cleaned_pages)}")
    return cleaned_pages

def add_metadata(clean_pages):
    metadata_list = []
    for i, page in enumerate(clean_pages):
        metadata = {
            "page_number": i + 1,
            "content_length": len(page),
            "summary": summarise_page(page)
        }
        metadata_list.append(metadata)
    return metadata_list

if __name__ == "__main__":
    book = '100M Lost Chapters by Alex Hormozi.txt'
    try:
        clean_pages = split_text(book)
    except FileNotFoundError:
        print(f"Error: The file '{book}' was not found. Due to copyright restrictions, please provide your own text file.")
        
    metadata_list = add_metadata(clean_pages)  # Process only the first 3 pages for testing
    with open('metadata.json', 'w', encoding='utf-8') as f:
        import json
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)
    # collection = initialize_chroma(clean_pages, metadata_list)

    # results = collection.query(query_texts=["What is the main theme of the book?"], n_results=3)
    # print("Query Results:")
    # for doc in results['documents'][0]:
    #     print(doc)


    