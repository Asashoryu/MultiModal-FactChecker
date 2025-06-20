from downloader import YouTubeAudioDownloader
from transcriber import AudioTranscriber
from pdf_processor import PDFProcessor
import whisper
import torch
import json
import os
from esg_summary import extract_table_metadata_with_summary, extract_image_metadata_with_summary


# Set paths
DATA_FOLDER = "data"
TRANSCRIPTIONS_FOLDER = "transcriptions"
ESG_REPORT_PATH = "data/Global_ESG_Flows_Q1_2024_Report.pdf"
IMAGE_FOLDER = "data/images"
os.makedirs(TRANSCRIPTIONS_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Initialize downloader
downloader = YouTubeAudioDownloader(output_folder=DATA_FOLDER)

# List of video URLs
video_urls = [
    "https://www.youtube.com/watch?v=qP1JKWBBy80",
    "https://www.youtube.com/watch?v=_p58cZIHDG4"
]

# Download audios, skipping existing files
audio_files = downloader.download_multiple_audios(video_urls)

# Initialize transcriber
transcriber = AudioTranscriber(input_folder=DATA_FOLDER)

# audio_data is directly returned from transcribe_all_audios()
audio_data = transcriber.transcribe_all_audios(audio_files)

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
transcriber.whisper_model = whisper.load_model("tiny", device=device)

# Transcribe audio files
transcriptions_dict = transcriber.transcribe_all_audios(audio_files)

# Save transcriptions
output_transcription_path = os.path.join(TRANSCRIPTIONS_FOLDER, "transcriptions.json")
with open(output_transcription_path, "w", encoding="utf-8") as json_file:
    json.dump(transcriptions_dict, json_file, indent=2)

print(f"Transcriptions saved to: {output_transcription_path}")

# Initialize PDF Processor
pdf_processor = PDFProcessor(ESG_REPORT_PATH, IMAGE_FOLDER)

# Extract structured data
raw_data = pdf_processor.extract_raw_data()
text_data = pdf_processor.extract_text_with_metadata()
image_data = pdf_processor.extract_image_metadata()
table_data = pdf_processor.extract_table_metadata()

# Save extracted ESG report data
with open(os.path.join(TRANSCRIPTIONS_FOLDER, "esg_text.json"), "w") as f:
    json.dump(text_data, f, indent=2)
with open(os.path.join(TRANSCRIPTIONS_FOLDER, "esg_images.json"), "w") as f:
    json.dump(image_data, f, indent=2)
with open(os.path.join(TRANSCRIPTIONS_FOLDER, "esg_tables.json"), "w") as f:
    json.dump(table_data, f, indent=2)

print(f"ESG Report text, images, and tables saved.")



# Display extracted images
#pdf_processor.display_images(image_data)

#THIS PART BELOW USES THE esg_summary.py script

# Extract & summarize ESG tables
table_summary_data = extract_table_metadata_with_summary(raw_data, ESG_REPORT_PATH)

# Extract & summarize ESG images
image_summary_data = extract_image_metadata_with_summary(raw_data, ESG_REPORT_PATH)

# Save summarized tables & images
with open(os.path.join(TRANSCRIPTIONS_FOLDER, "esg_table_summary.json"), "w") as f:
    json.dump(table_summary_data, f, indent=2)

with open(os.path.join(TRANSCRIPTIONS_FOLDER, "esg_image_summary.json"), "w") as f:
    json.dump(image_summary_data, f, indent=2)

print("ESG table and image summaries saved successfully.")


#I made the main script to follow these steps so that we don't have to re run things i.e:

# 1. Model Download (Flan-T5)
# Will NOT redownload the Flan-T5 model unless:You delete Hugging Face cache (~/.cache/huggingface).


# 2. YouTube Videos
# Will NOT redownload YouTube videos that already exist in data/. If a video exists, it prints: "Skipping [filename], already downloaded." If a video does not exist, it will download it.

# 3. Audio Transcription
# Always runs transcription on the downloaded audio files.

# 4. ESG Report Processing
# Always runs text, table, and image extraction from the PDF.

# 5. ESG Summarization
# Always generates summaries for extracted tables and images.

# 6. JSON Data Saving
# Always saves transcriptions, extracted ESG text, images, and tables into JSON files.




# THIS PART IS ABOUT DATA STORAGE
from weaviate_vector_storage import ingest_all_data
from weaviate_vector_storage import reset_collection

# Reset Weaviate collection before re-ingesting data
reset_collection()

# Ingest multimodal data into Weaviate
ingest_all_data("RAGESGDocuments", audio_data, text_data, image_data, table_data)



# THIS PART
from esg_analysis import analyze_and_print_esg_results

# Example queries
user_question_1 = "Is ESG investment a fraud?"
user_question_2 = "How did European sustainable fund flows perform in Q1 2024 compared to the previous quarter?"
user_question_3 = "What is the net flows for Parnassus Mid Cap Fund?"

# Run analysis
analyze_and_print_esg_results(user_question_1)
analyze_and_print_esg_results(user_question_2)
analyze_and_print_esg_results(user_question_3)
