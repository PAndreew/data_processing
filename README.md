# Invoice Extraction Pipeline

## Overview

This project is a modular, configurable pipeline for automated invoice data extraction from PDF files. It is designed to be robust, extensible, and suitable for both cloud and on-premises deployments.

The pipeline processes batches of PDF invoices, converts relevant pages to images, and extracts structured data using dual Large Language Models (LLMs) for increased reliability. Results are saved in CSV format, and detailed logs are maintained for traceability.

## Features

- **Modular Design:** Each step (PDF conversion, image processing, LLM extraction, data comparison, logging) is implemented as a separate module for easy maintenance and extension.
- **Configurable:** All major parameters (input/output folders, LLM settings, retry logic, logging level, etc.) can be set via command-line arguments or environment variables.
- **Dual LLM Validation:** Uses two independent LLMs to extract and cross-validate invoice data, improving accuracy and reliability.
- **Cloud & On-Prem Ready:** The pipeline can be run locally or deployed to cloud environments with minimal changes.
- **Error Handling & Logging:** Comprehensive error handling and logging for each step, with automatic cleanup of temporary files.
- **Extensible Schema:** Uses a Pydantic model for invoice data, making it easy to adapt to new fields or formats.

## Usage

1. **Install Dependencies**
   - Install required Python packages:
     ```
     pip install -r requirements.txt
     ```

2. **Prepare Environment**
   - Set up environment variables for LLM API keys and endpoints (see `.env` file or pass via CLI).
    ```env
    GOOGLE_API_KEY = "secretapikey"
    OPENAI_SECRET_KEY = "secretkey"

    OPENAI_BASE_URL = "https://api.openai.com/v1"
    GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    ```

3. **Run the Pipeline**
   - Example command:
     ```
     python main_pipeline.py <input_folder> -o <output_folder> --llm1_api_key <key> --llm1_base_url <url> --llm2_api_key <key> --llm2_base_url <url>
     ```
   - See `main_pipeline.py --help` for all options.

4. **Results**
   - Extracted invoice data is saved as a CSV in the output folder.
   - Logs and temporary files are managed automatically.

## Modules

- `convert_pdf_to_image.py`: Converts first and last pages of PDFs to JPG images.
- `extract_invoice_data_async.py`: Extracts structured invoice data from images using two LLMs and compares results.
- `main_pipeline.py`: Orchestrates the end-to-end process, manages configuration, logging, and output.

## Configuration

- **LLM Providers:** Supports multiple LLMs (e.g., Gemini, OpenAI) via configurable API keys and endpoints.
- **Retry Logic:** Automatic retries with exponential backoff if LLMs disagree or fail.
- **Logging:** Adjustable log levels and output locations.

## Deployment

- **On-Premises:** Run as a Python script on local servers.
- **Cloud:** Can be containerized and deployed to cloud platforms (e.g., Azure, AWS, GCP) with environment-based configuration.

## Roadmap

- **Short Term**
  - Add support for more invoice formats and languages.
  - Improve LLM prompt engineering for higher accuracy.
  - Enhance error reporting and diagnostics.

- **Medium Term**
  - Integrate with cloud storage (e.g., S3, Azure Blob) for input/output.
  - Add web-based dashboard for monitoring and configuration.
  - Support batch processing and parallelization.

- **Long Term**
  - Implement active learning for continuous improvement.
  - Add support for additional document types (e.g., receipts, contracts).
  - Provide REST API and microservice deployment options.