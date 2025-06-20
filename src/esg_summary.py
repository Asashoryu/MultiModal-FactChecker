from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import pipeline
import os
import base64
from unstructured.documents.elements import Table

# Table summarization prompt
TABLES_SUMMARIZER_PROMPT = """
As an ESG analyst for emerging markets investments, provide a concise summary of the table.
Focus on ESG metrics, trends, comparisons, or outliers relevant to emerging markets.
Ensure it's precise and informative.

Table: {table_content}

Limit your summary to 3-4 sentences.
"""

# Image summarization prompt
IMAGES_SUMMARIZER_PROMPT = """
As an ESG analyst for emerging markets investments, describe key insights from the image.
Focus on ESG-relevant content and its emerging market context.
Deliver a coherent summary that captures the image's essence.

Image: {image_element}

Limit your description to 3-4 sentences.
"""

# Initialize text generation pipeline
text_generator = pipeline("text2text-generation", model="google/flan-t5-base")

# def generate_llm_response(prompt: str) -> str:
#     """Generates a response from an LLM based on the provided prompt."""
#     response = text_generator(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
#     return response

def generate_llm_response(prompt: str) -> str:
    """Generates a response from an LLM with a safe token limit."""
    # Ensure prompt is within model's max token limit
    prompt = prompt[:512]  # Truncate to 512 tokens because i noticed that the Flan-T5 model has a maximum token limit of 512
    
    response = text_generator(prompt, max_new_tokens=512, do_sample=False)[0]['generated_text']
    return response


def extract_table_metadata_with_summary(esg_report, source_document):
    """Extracts tables and summarizes them using an LLM."""
    table_data = []
    prompt_template = ChatPromptTemplate.from_template(TABLES_SUMMARIZER_PROMPT)

    for element in esg_report:
        if isinstance(element, Table):
            page_number = element.metadata.page_number
            table_content = str(element)

            # Format prompt using LangChain template
            messages = prompt_template.format_messages(table_content=table_content)
            full_prompt = messages[0].content if messages else ""

            # Generate summary using LLM
            description = generate_llm_response(full_prompt)

            table_data.append({
                "source_document": source_document,
                "page_number": page_number,
                "table_content": table_content,
                "description": description.strip()
            })
    return table_data

def extract_image_metadata_with_summary(esg_report, source_document):
    """Extracts image metadata and summarizes them using an LLM."""
    image_data = []
    prompt_template = ChatPromptTemplate.from_template(IMAGES_SUMMARIZER_PROMPT)

    for element in esg_report:
        if "Image" in str(type(element)):
            page_number = getattr(element.metadata, 'page_number', None)
            image_path = getattr(element.metadata, 'image_path', None)

            if image_path and os.path.exists(image_path):
                # Format prompt
                messages = prompt_template.format_messages(image_element=image_path)
                full_prompt = messages[0].content if messages else ""
                description = generate_llm_response(full_prompt)

                with open(image_path, "rb") as img_file:
                    encoded_string = base64.b64encode(img_file.read()).decode("utf-8")

                image_data.append({
                    "source_document": source_document,
                    "page_number": page_number,
                    "image_path": image_path,
                    "description": description.strip(),
                    "base64_encoding": encoded_string
                })
            else:
                print(f"Warning: Image file not found for page {page_number}")

    return image_data
