import os
import math
import json
import matplotlib.pyplot as plt
from PIL import Image
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import NarrativeText, Image, Table

class PDFProcessor:
    def __init__(self, pdf_path, output_image_dir):
        self.pdf_path = pdf_path
        self.output_image_dir = output_image_dir
        os.makedirs(self.output_image_dir, exist_ok=True)
        self.raw_data = None

    def extract_raw_data(self):
        """Partition PDF into structured elements."""
        self.raw_data = partition_pdf(
            filename=self.pdf_path,
            strategy="hi_res",
            extract_images_in_pdf=True,
            extract_image_block_to_payload=False,
            extract_image_block_output_dir=self.output_image_dir
        )
        return self.raw_data

    def extract_text_with_metadata(self):
        """Extract structured text with metadata."""
        text_data = []
        paragraph_counters = {}

        for element in self.raw_data:
            if isinstance(element, NarrativeText):
                page_number = element.metadata.page_number

                if page_number not in paragraph_counters:
                    paragraph_counters[page_number] = 1
                else:
                    paragraph_counters[page_number] += 1

                paragraph_number = paragraph_counters[page_number]

                text_data.append({
                    "source_document": self.pdf_path,
                    "page_number": page_number,
                    "paragraph_number": paragraph_number,
                    "text": element.text
                })

        return text_data

    def extract_image_metadata(self):
        """Extract image metadata from the report."""
        image_data = []
        for element in self.raw_data:
            if "Image" in str(type(element)):
                image_data.append({
                    "source_document": self.pdf_path,
                    "page_number": element.metadata.page_number,
                    "image_path": getattr(element.metadata, "image_path", None)
                })
        return image_data

    def display_images(self, extracted_image_data, images_per_row=4):
        """Display extracted images in a grid format."""
        valid_images = [img for img in extracted_image_data if img['image_path']]
        if not valid_images:
            print("No valid image data available.")
            return

        num_images = len(valid_images)
        num_rows = math.ceil(num_images / images_per_row)
        fig, axes = plt.subplots(num_rows, images_per_row, figsize=(20, 5 * num_rows))
        axes = axes.flatten() if num_rows > 1 else [axes]

        for ax, img_data in zip(axes, valid_images):
            try:
                img = Image.open(img_data['image_path'])
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f"Page {img_data['page_number']}", fontsize=10)
            except Exception as e:
                print(f"Error loading image {img_data['image_path']}: {str(e)}")
                ax.text(0.5, 0.5, "Error loading image", ha='center', va='center')
                ax.axis('off')

        for ax in axes[num_images:]:
            fig.delaxes(ax)

        plt.tight_layout()
        plt.show()

    def extract_table_metadata(self):
        """Extract tables from the ESG report."""
        table_data = []
        for element in self.raw_data:
            if isinstance(element, Table):
                table_data.append({
                    "source_document": self.pdf_path,
                    "page_number": element.metadata.page_number,
                    "table_content": str(element)
                })
        return table_data
