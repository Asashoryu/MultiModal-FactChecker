import textwrap
from vector_storage import search_multimodal
# from esg_summary import generate_response
from esg_summary import generate_llm_response

def esg_analysis(user_query: str):
    """Retrieve ESG documents from ChromaDB and assemble context for AI response."""
    search_results = search_multimodal(user_query)

    context = ""  # Start assembling the context
    sources = []

    # Ensure we process results correctly
    if search_results and "metadatas" in search_results and search_results["metadatas"]:
        for item in search_results["metadatas"][0]:  # ChromaDB stores metadata in lists
            ctype = item.get("content_type", "unknown")

            if ctype == "audio":
                context += f"Audio Transcription from {item['url']}: {item['transcription']}\n\n"
            elif ctype == "text":
                context += f"Text from {item['source_document']} (Page {item['page_number']}, Paragraph {item['paragraph_number']}): {item['text']}\n\n"
            elif ctype == "image":
                context += f"Image from {item['source_document']} (Page {item['page_number']}, Path: {item['image_path']})\n\n"
            elif ctype == "table":
                context += f"Table from {item['source_document']} (Page {item['page_number']}): {item['table_content']}\n\n"

            sources.append(item)  # Store metadata for reference

    # Fallback: If no results, provide a default message
    if not context.strip():
        context = "No relevant ESG documents found for this query."

    response = generate_llm_response(user_query)

    return {
        "user_query": user_query,
        "ai_response": response,
        "sources": sources
    }


def wrap_text(text, width=120):
    """Wraps text for better readability."""
    return textwrap.fill(text, width=width)

def analyze_and_print_esg_results(user_question):
    """Runs ESG analysis and prints structured results."""
    result = esg_analysis(user_question)

    print("User Query:", result["user_query"])
    print("\nAI Response:", wrap_text(result["ai_response"]))
    print("\nSources (sorted by relevance):")
    for source in result["sources"]:
        print(f"- Type: {source['type']}, Distance: {source['distance']:.3f}")
        if source['type'] == 'text':
            print(f" Document: {source['document']}, Page: {source['page']}, Paragraph: {source['paragraph']}")
        elif source['type'] == 'image':
            print(f" Document: {source['document']}, Page: {source['page']}, Image Path: {source['image_path']}")
        elif source['type'] == 'table':
            print(f" Document: {source['document']}, Page: {source['page']}")
        elif source['type'] == 'audio':
            print(f" URL: {source['url']}")
        print("---")
