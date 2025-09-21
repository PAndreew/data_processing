# config.py
"""
Configuration file for the Batch Audit Assistant Streamlit application.
This file centralizes all the tunable parameters, model names, and prompts
for easy management and modification.
"""
from pathlib import Path

# --- LLM & AI SETTINGS ---
# A powerful model for the complex task of document segmentation.
LLM_SEGMENTER_MODEL = "gemini-2.5-pro" # Note: Changed to a known stable model name
# Two identical, fast, and cheap models for the consensus-based extraction approach.
LLM1_MODEL_PRIMARY = "gemini-2.5-flash"
LLM2_MODEL_VERIFIER = "gemini-2.5-flash"

# --- PERFORMANCE SETTINGS ---
# --- OPTIMIZATION 3: Make DPI configurable ---
IMAGE_DPI = 200  # Dots Per Inch for PDF to image conversion. 200 is a good balance of quality and speed.

# Fixed concurrency and delay settings for performance and rate-limit safety
CONCURRENCY_LIMIT = 10  # Process up to 10 PDFs at the same time
INTER_PDF_DELAY_S = 0  # No delay between tasks

# --- APPLICATION PATHS & CREDENTIALS ---
# Directory for storing persistent application data, like credentials.
APP_DIR = Path.home() / ".invoice_audit_app"
CREDENTIALS_FILE = APP_DIR / "data_processor.json"

# --- GOOGLE CLOUD SETTINGS (HARDCODED) ---
# The GCP Project and Location are fixed for this application.
GCP_PROJECT_ID = "enttree"
GCP_LOCATION = "us-central1"

# --- AI PROMPT TEMPLATES ---
# Note: These are templates. The final prompts are formatted in app.py with the Pydantic schemas.

# For Extraction
SYSTEM_PROMPT_FOR_EXTRACTION_TEMPLATE = """You are an expert invoice data extraction system. \
Your response MUST be a single, valid JSON object that strictly adheres to the InvoiceData schema. \
Do not include markdown or any other text outside the JSON object. \
Schema: {schema}"""

USER_PROMPT_FOR_EXTRACTION = "Extract all fields from the provided invoice image(s)."

RETRY_USER_PROMPT_TEMPLATE = """A previous attempt by two LLMs resulted in a mismatch. \
Re-examine the image(s) and previous outputs to generate the definitive, correct JSON. \
LLM1 produced: {llm1_previous_output_str}. \
LLM2 produced: {llm2_previous_output_str}."""

# For Segmentation
SYSTEM_PROMPT_FOR_SEGMENTATION_TEMPLATE = """You are an expert document analysis system. \
Your task is to identify and group pages that belong to distinct, individual invoices within a document.
For each distinct invoice, you must specify its 0-indexed `start_page` and `end_page`.
Ignore any pages that are not part of an invoice (e.g., cover letters, contracts, detailed accounts for utilities, etc).
Invoices usually start with the word 'Számla' and usually end with words like 'Összesen', 'Fizetendő'.
Your response MUST be a single, valid JSON object strictly adhering to the DocumentAnalysis schema.
Schema: {schema}"""

USER_PROMPT_FOR_SEGMENTATION = """Analyze the provided document images and identify all distinct invoices. \
For each separate invoice you find, provide its start and end page numbers (0-indexed). \
If no invoices are found, return an empty list for 'invoices'."""