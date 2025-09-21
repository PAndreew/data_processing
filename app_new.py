# app.py
# --- IMPORTS ---
# Standard Library
import asyncio
import base64
import json
import logging
import sys
import os
import io
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, IO

# Third-party Libraries
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from openpyxl import Workbook

# --- CONFIG IMPORT ---
# Create a dummy config.py if you don't have one to run this script
if not os.path.exists("config.py"):
    with open("config.py", "w") as f:
        f.write('GCP_PROJECT_ID = "your-gcp-project-id"\n')
        f.write('GCP_LOCATION = "us-central1"\n')
        f.write('APP_DIR = Path(".")\n')
        f.write('CREDENTIALS_FILE = APP_DIR / "credentials.json"\n')
        f.write('IMAGE_DPI = 150\n')
        f.write('LLM1_MODEL_PRIMARY = "gemini-1.5-pro-001"\n')
        f.write('LLM2_MODEL_VERIFIER = "gemini-1.5-pro-001"\n')
        f.write('LLM_SEGMENTER_MODEL = "gemini-1.5-pro-001"\n')
        f.write('CONCURRENCY_LIMIT = 5\n')
        f.write('INTER_PDF_DELAY_S = 1\n')
        f.write('SYSTEM_PROMPT_FOR_EXTRACTION_TEMPLATE = "Extract data based on this schema: {schema}"\n')
        f.write('USER_PROMPT_FOR_EXTRACTION = "Extract invoice data."\n')
        f.write('RETRY_USER_PROMPT_TEMPLATE = "Retry extraction. LLM1 said: {llm1_previous_output_str}, LLM2 said: {llm2_previous_output_str}"\n')
        f.write('SYSTEM_PROMPT_FOR_SEGMENTATION_TEMPLATE = "Segment the document based on this schema: {schema}"\n')
        f.write('USER_PROMPT_FOR_SEGMENTATION = "Find all invoices in the document."\n')
import config

# --- GOOGLE CLOUD / VERTEX AI IMPORTS (with fallback for local UI testing) ---
# try:
import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from google.api_core import exceptions as google_exceptions
from google.auth import exceptions as google_auth_exceptions
from google.oauth2 import service_account
VERTEXAI_AVAILABLE = True
# except ImportError:
#     VERTEXAI_AVAILABLE = False
#     class GenerativeModel: pass
#     class Part: pass
#     class GenerationConfig: pass
#     class google_exceptions: pass
#     class google_auth_exceptions: pass
#     class DefaultCredentialsError(Exception): pass
#     class vertexai:
#         @staticmethod
#         def init(*args, **kwargs):
#             logging.warning("Vertex AI SDK not found. PDF extraction will fail.")
#             raise google_auth_exceptions.DefaultCredentialsError("SDK not found")


# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# --- CREDENTIALS MANAGEMENT ---

def get_gcp_credentials_from_base64():
    """
    Decodes the Base64 encoded GCP credentials from Streamlit secrets.
    Returns a credentials object.
    """
    # Get the base64 encoded string from secrets
    creds_base64 = st.secrets["gcp_creds_base64"]
    
    # Decode the base64 string into bytes, then into a JSON string
    creds_json_bytes = base64.b64decode(creds_base64)
    creds_json_str = creds_json_bytes.decode('utf-8')
    
    # Load the JSON string into a dictionary
    creds_dict = json.loads(creds_json_str)
    
    # Create and return credentials object from the dictionary
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    return credentials


def initialize_vertex_ai():
    """
    Initializes Vertex AI based on the environment.
    - On Streamlit Cloud, uses the Base64 secret.
    - Locally, uses the credentials.json file.
    Returns True if authentication is successful, False otherwise.
    """
    if not VERTEXAI_AVAILABLE:
        st.error("Vertex AI SDK is not installed. PDF extraction is disabled.")
        return False
        
    try:
        # Check if running on Streamlit Cloud with the secret configured
        if hasattr(st, 'secrets') and "gcp_creds_base64" in st.secrets:
            logger.info("Authenticating via Streamlit secrets...")
            credentials = get_gcp_credentials_from_base64()
            vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION, credentials=credentials)
            # st.success(f"✅ Authenticated to GCP Project: **{config.GCP_PROJECT_ID}**")
            return True
            
        # Fallback to local file-based authentication
        elif config.CREDENTIALS_FILE.exists():
            logger.info(f"Authenticating via local file: {config.CREDENTIALS_FILE}")
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(config.CREDENTIALS_FILE)
            vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
            # st.success(f"✅ Authenticated to GCP Project: **{config.GCP_PROJECT_ID}**")
            return True
            
        else:
            # No credentials found in either environment
            logger.warning("No credentials found. Waiting for user upload.")
            st.warning(f"🚫 Nem tudtunk kapcsolódni a Gemini szolgáltatáshoz!**{config.GCP_PROJECT_ID}**")
            return False

    except Exception as e:
        st.error(f"Authentication failed: {e}")
        logger.error(f"Authentication failed with error: {e}", exc_info=True)
        return False


def save_credentials(uploaded_file):
    """Saves the uploaded credentials file locally."""
    if uploaded_file:
        try:
            config.APP_DIR.mkdir(exist_ok=True)
            with open(config.CREDENTIALS_FILE, "wb") as f:
                f.write(uploaded_file.getbuffer())
            logger.info(f"Credentials saved to {config.CREDENTIALS_FILE}")
            # The app will rerun, and initialize_vertex_ai() will now find the file.
            return True
        except Exception as e:
            st.error(f"Error saving credentials: {e}")
            return False
    return False


# --- Pydantic Models for Data Validation ---
class InvoiceData(BaseModel):
    seller_name: Optional[str] = Field(None, description="Az eladó cégneve")
    buyer_name: Optional[str] = Field(None, description="A vevő cégneve")
    invoice_number: Optional[str] = Field(None, description="A számla egyedi sorszáma")
    invoice_date: Optional[str] = Field(None, description="A számla kiállítási dátuma ÉÉÉÉ-HH-NN formátumban")
    total_gross_amount: Optional[float] = Field(None, description="A számla teljes bruttó összege")
    total_net_amount: Optional[float] = Field(None, description="A számla teljes nettó összege")
    total_vat_amount: Optional[float] = Field(None, description="A számla teljes ÁFA összege")
    currency: Optional[str] = Field(None, description="A használt pénznem (pl. HUF, EUR)")
    seller_tax_id: Optional[str] = Field(None, description="Az eladó adószáma")
    buyer_tax_id: Optional[str] = Field(None, description="A vevő adószáma")
    exchange_rate: Optional[float] = Field(1.0, description="Az árfolyam, alapértelmezetten 1.0")

class InvoiceSegment(BaseModel):
    description: Optional[str] = Field(None, description="A brief one-line description of the invoice")
    start_page: int = Field(description="The 0-indexed starting page number")
    end_page: int = Field(description="The 0-indexed ending page number")

class DocumentAnalysis(BaseModel):
    invoices: List[InvoiceSegment] = Field(description="A list of all distinct invoices found")


# --- PDF EXTRACTION LOGIC (TAB 1) ---

# --- AI PROMPTS ---
SYSTEM_PROMPT_FOR_EXTRACTION = config.SYSTEM_PROMPT_FOR_EXTRACTION_TEMPLATE.format(schema=json.dumps(InvoiceData.model_json_schema(), indent=2))
USER_PROMPT_FOR_EXTRACTION = config.USER_PROMPT_FOR_EXTRACTION
RETRY_USER_PROMPT_TEMPLATE = config.RETRY_USER_PROMPT_TEMPLATE.format(llm1_previous_output_str="{llm1_previous_output_str}", llm2_previous_output_str="{llm2_previous_output_str}")
SYSTEM_PROMPT_FOR_SEGMENTATION = config.SYSTEM_PROMPT_FOR_SEGMENTATION_TEMPLATE.format(schema=json.dumps(DocumentAnalysis.model_json_schema(), indent=2))
USER_PROMPT_FOR_SEGMENTATION = config.USER_PROMPT_FOR_SEGMENTATION

def write_to_excel(all_data: List[Dict[str, Any]], filename: str):
    """Writes extracted data to a single-sheet Excel file."""
    if not all_data:
        logger.warning("No data provided to write to Excel.")
        return
    df = pd.DataFrame(all_data)
    preferred_order = [
        'source_filename', 'invoice_number', 'invoice_date', 'seller_name', 'seller_tax_id',
        'buyer_name', 'buyer_tax_id', 'total_gross_amount', 'total_net_amount',
        'total_vat_amount', 'currency', 'exchange_rate', 'start_page', 'end_page'
    ]
    existing_cols = [col for col in preferred_order if col in df.columns]
    remaining_cols = sorted([col for col in df.columns if col not in existing_cols])
    df = df[existing_cols + remaining_cols]
    df.to_excel(filename, index=False, engine='openpyxl')
    logger.info(f"Successfully saved {len(all_data)} records to {filename}.")

def _sync_convert_pdf_to_jpg_bytes(pdf_bytes: bytes, pdf_filename: str) -> List[bytes]:
    images = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=config.IMAGE_DPI, alpha=False)
            images.append(pix.tobytes("jpeg"))
        doc.close()
    except Exception as e:
        logger.error(f"Failed to convert PDF '{pdf_filename}': {e}", exc_info=True)
    return images

async def convert_pdf_to_jpg_bytes_async(pdf_bytes: bytes, pdf_filename: str) -> List[bytes]:
    return await asyncio.to_thread(_sync_convert_pdf_to_jpg_bytes, pdf_bytes, pdf_filename)

async def _extract_with_vertex_ai(model_name: str, image_bytes: List[bytes], user_prompt: str) -> Tuple[Optional[InvoiceData], str]:
    try:
        model = GenerativeModel(model_name, system_instruction=[SYSTEM_PROMPT_FOR_EXTRACTION])
        content = [user_prompt] + [Part.from_data(img, mime_type="image/jpeg") for img in image_bytes]
        generation_config = GenerationConfig(response_mime_type="application/json")
        response = await model.generate_content_async(content, generation_config=generation_config)
        return InvoiceData.model_validate_json(response.text), response.text
    except Exception as e:
        logger.error(f"API call to {model_name} failed: {e}", exc_info=True)
        return None, json.dumps({"error": str(e)})

async def _find_invoices_in_document(image_bytes: List[bytes]) -> Optional[DocumentAnalysis]:
    try:
        model = GenerativeModel(config.LLM_SEGMENTER_MODEL, system_instruction=[SYSTEM_PROMPT_FOR_SEGMENTATION])
        content = [USER_PROMPT_FOR_SEGMENTATION] + [Part.from_data(img, mime_type="image/jpeg") for img in image_bytes]
        generation_config = GenerationConfig(response_mime_type="application/json")
        response = await model.generate_content_async(content, generation_config=generation_config)
        return DocumentAnalysis.model_validate_json(response.text)
    except Exception as e:
        logger.error(f"Segmentation call failed: {e}", exc_info=True)
        st.session_state.log.append(f"CRITICAL: Document segmentation failed: {e}")
        return None

def _compare_results(data1: Optional[InvoiceData], data2: Optional[InvoiceData]) -> bool:
    if not data1 or not data2: return False
    return data1.model_dump() == data2.model_dump()

async def _run_consensus_extraction(image_page_images: List[bytes], pdf_filename: str, log_context: str) -> Tuple[Optional[InvoiceData], str]:
    log_prefix = f"[{pdf_filename} - {log_context}]"
    last_raw1, last_raw2 = "N/A", "N/A"
    for attempt in range(2):
        prompt = (USER_PROMPT_FOR_EXTRACTION if attempt == 0 else RETRY_USER_PROMPT_TEMPLATE.format(llm1_previous_output_str=last_raw1, llm2_previous_output_str=last_raw2))
        tasks = [
            _extract_with_vertex_ai(config.LLM1_MODEL_PRIMARY, image_page_images, prompt),
            _extract_with_vertex_ai(config.LLM2_MODEL_VERIFIER, image_page_images, prompt)
        ]
        (data1, raw1), (data2, raw2) = await asyncio.gather(*tasks)
        last_raw1, last_raw2 = raw1, raw2
        st.session_state.log.append(f"INFO: {log_prefix} Attempt {attempt + 1} LLM 1 Output:\n{raw1}")
        st.session_state.log.append(f"INFO: {log_prefix} Attempt {attempt + 1} LLM 2 Output:\n{raw2}")
        if _compare_results(data1, data2):
            st.session_state.log.append(f"SUCCESS: {log_prefix} Success on attempt {attempt + 1}.")
            return data1, "Success"
        else:
            st.session_state.log.append(f"WARN: {log_prefix} Mismatch on attempt {attempt + 1}.")
    st.session_state.log.append(f"ERROR: {log_prefix} Failed after all attempts.")
    return None, "Failed (Mismatch)"

async def process_single_pdf_task(pdf_file: IO[bytes], status_placeholder) -> List[Dict]:
    pdf_filename = pdf_file.name
    status_placeholder.markdown(f"**{pdf_filename}** | ⏳ Converting...")
    st.session_state.log.append(f"INFO: [{pdf_filename}] Starting processing.")
    all_page_images = await convert_pdf_to_jpg_bytes_async(pdf_file.read(), pdf_filename)
    if not all_page_images:
        status_placeholder.markdown(f"**{pdf_filename}** | ❌ Failed (PDF Error)")
        return [{"filename": pdf_filename, "status": "Failed (PDF Error)", "data": None}]

    status_placeholder.markdown(f"**{pdf_filename}** | ⏳ Tier 1: Fast extraction...")
    fast_path_data, fast_path_status = await _run_consensus_extraction(all_page_images, pdf_filename, "Fast Path")
    if fast_path_status == "Success" and fast_path_data:
        status_placeholder.markdown(f"**{pdf_filename}** | ✅ Success (Fast Path)")
        result_data = fast_path_data.model_dump()
        result_data['source_filename'] = pdf_filename
        return [{"filename": pdf_filename, "status": "Success", "data": result_data}]

    status_placeholder.markdown(f"**{pdf_filename}** | ⏳ Tier 2: Segmenting document...")
    analysis_result = await _find_invoices_in_document(all_page_images)
    if not analysis_result or not analysis_result.invoices:
        status_msg = "✅ (No Invoices)" if analysis_result else "❌ (Segmentation Failed)"
        status_placeholder.markdown(f"**{pdf_filename}** | {status_msg}")
        return [{"filename": pdf_filename, "status": status_msg, "data": None}]

    num_invoices = len(analysis_result.invoices)
    status_placeholder.markdown(f"**{pdf_filename}** | ⏳ Extracting {num_invoices} invoice(s)...")
    processed_results, success_count = [], 0
    for i, segment in enumerate(analysis_result.invoices):
        log_context = f"Segment {i+1} (Pages: {segment.start_page}-{segment.end_page})"
        try:
            invoice_pages = [all_page_images[p] for p in range(segment.start_page, segment.end_page + 1)]
        except IndexError:
            st.session_state.log.append(f"ERROR: [{pdf_filename}] Invalid page range. Skipping.")
            continue
        extracted_data, status = await _run_consensus_extraction(invoice_pages, pdf_filename, log_context)
        result = {"filename": pdf_filename, "status": status, "data": None}
        if status == "Success" and extracted_data:
            success_count += 1
            result_data = extracted_data.model_dump()
            result_data.update({'source_filename': pdf_filename, 'start_page': segment.start_page, 'end_page': segment.end_page})
            result["data"] = result_data
        processed_results.append(result)

    emoji = "✅" if success_count == num_invoices else "⚠️" if success_count > 0 else "❌"
    status_placeholder.markdown(f"**{pdf_filename}** | {emoji} Done ({success_count}/{num_invoices})")
    return processed_results

async def run_task_with_inter_pdf_delay(semaphore: asyncio.Semaphore, *args, **kwargs):
    async with semaphore:
        result = await process_single_pdf_task(*args, **kwargs)
        if config.INTER_PDF_DELAY_S > 0:
            await asyncio.sleep(config.INTER_PDF_DELAY_S)
        return result

async def run_batch_extraction(pdf_files):
    try:
        vertexai.init(project=config.GCP_PROJECT_ID, location=config.GCP_LOCATION)
    except (google_auth_exceptions.DefaultCredentialsError, Exception) as e:
        st.error(f"Authentication failed: {e}")
        return

    semaphore = asyncio.Semaphore(config.CONCURRENCY_LIMIT)
    status_placeholders = {f.name: st.empty() for f in pdf_files}
    tasks = [run_task_with_inter_pdf_delay(semaphore, pdf, status_placeholders[pdf.name]) for pdf in pdf_files]
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    successful_data = []
    for item in results_list:
        if isinstance(item, list):
            for result in item:
                if result and result.get('status') == 'Success' and result.get('data'):
                    successful_data.append(result['data'])
                    unique_key = f"{result['filename']}_{result['data'].get('invoice_number', 'N/A')}"
                    st.session_state.extracted_data[unique_key] = result
    
    if successful_data:
        filename = f"extraction_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        st.session_state.output_excel_file = filename
        write_to_excel(successful_data, filename)
    else:
        st.session_state.log.append("INFO: No successful extractions to save.")


# --- DATA COMPARISON LOGIC (TAB 2) ---

def load_processed_file(uploaded_file):
    """
    Loads the main 'processed' file, converts non-HUF amounts to HUF using the
    exchange rate, and ensures correct data types.
    """
    print("Loading Processed file...")
    try:
        df = pd.read_excel(uploaded_file)
        
        # Define required columns for validation
        required_cols = [
            'invoice_number', 'total_gross_amount', 
            'total_net_amount', 'total_vat_amount'
        ]
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_cols):
            st.error(f"Processed file is missing one of the required columns: {required_cols}")
            return pd.DataFrame()

        # Standardize key column types
        df['invoice_number'] = df['invoice_number'].astype(str).str.strip()
        for col in ['total_gross_amount', 'total_net_amount', 'total_vat_amount', 'exchange_rate']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- REQUIREMENT A: CURRENCY CONVERSION LOGIC ---
        # Ensure currency and exchange_rate columns exist for conversion
        if 'currency' in df.columns and 'exchange_rate' in df.columns:
            # Fill NaN exchange rates with 1.0 to avoid errors in multiplication
            df['exchange_rate'].fillna(1.0, inplace=True)
            
            # Identify rows that are not in HUF and have a valid exchange rate
            conversion_mask = (df['currency'].str.upper() != 'HUF') & (df['exchange_rate'].notna())
            
            # Apply exchange rate to amount columns for the selected rows
            amount_cols = ['total_gross_amount', 'total_net_amount', 'total_vat_amount']
            for col in amount_cols:
                if col in df.columns:
                    # Use .loc to ensure we are modifying the original dataframe slice
                    df.loc[conversion_mask, col] = df.loc[conversion_mask, col] * df.loc[conversion_mask, 'exchange_rate']
            
            print(f"Applied currency conversion to {conversion_mask.sum()} rows.")
        else:
            print("Warning: 'currency' or 'exchange_rate' column not found. Skipping currency conversion.")
        
        print("Success: Processed file loaded and amounts adjusted to HUF.")
        return df

    except Exception as e:
        st.error(f"Error loading the Processed file: {e}")
        return pd.DataFrame()

def load_nav_file(uploaded_file):
    """
    Loads the NAV file, standardizes its columns, and aggregates rows with
    the same invoice number by summing their amounts.
    """
    print("Loading NAV file...")
    try:
        df = pd.read_excel(uploaded_file, skiprows=5)
        
        COLUMN_MAP = {
            'számlasorszám': 'invoice_number',
            'bruttó érték Ft': 'total_gross_amount',
            'nettóérték Ft': 'total_net_amount',
            'adóérték Ft': 'total_vat_amount',
            'számla dátuma': 'invoice_date',
            'vevő megnevezése': 'buyer_name',
            'eladó megnevezése': 'seller_name'
        }
        df.rename(columns=COLUMN_MAP, inplace=True)

        if 'invoice_number' not in df.columns:
            st.error("NAV file must contain 'számlasorszám' column.")
            return pd.DataFrame()

        # Standardize key column types and drop rows with no invoice number
        df.dropna(subset=['invoice_number'], inplace=True)
        df['invoice_number'] = df['invoice_number'].astype(str).str.strip()
        
        numeric_cols = ['total_gross_amount', 'total_net_amount', 'total_vat_amount']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- REQUIREMENT B: AGGREGATION LOGIC ---
        agg_rules = {}
        standard_cols_to_keep = ['invoice_number']
        
        # Build aggregation rules dynamically based on available columns
        for standard_name in COLUMN_MAP.values():
            if standard_name in df.columns and standard_name != 'invoice_number':
                standard_cols_to_keep.append(standard_name)
                if standard_name in numeric_cols:
                    agg_rules[standard_name] = 'sum'
                else:
                    agg_rules[standard_name] = 'first' # Keep the first value for non-numeric fields

        if not df.empty:
            print(f"NAV file rows before aggregation: {len(df)}")
            df = df.groupby('invoice_number').agg(agg_rules).reset_index()
            print(f"NAV file rows after aggregation: {len(df)}")
        
        print("Success: NAV file loaded, standardized, and aggregated.")
        return df[standard_cols_to_keep]

    except Exception as e:
        st.error(f"Error loading the NAV file: {e}")
        return pd.DataFrame()

def load_karton_file(uploaded_file):
    """
    Loads the Karton file, standardizes its columns, and aggregates rows with
    the same invoice number by summing amounts.
    """
    print("Loading Karton file...")
    try:
        df = pd.read_excel(uploaded_file)

        COLUMN_MAP = {
            'hivatkozas': 'invoice_number',
            'tartozik': 'total_gross_amount',
            'datum': 'invoice_date',
            'partnev': 'partner_name'
        }
        df.rename(columns=COLUMN_MAP, inplace=True)

        if 'invoice_number' not in df.columns:
            st.error("Karton file must contain 'hivatkozas' column.")
            return pd.DataFrame()

        # Standardize key column types and drop rows with no invoice number
        df.dropna(subset=['invoice_number'], inplace=True)
        df['invoice_number'] = df['invoice_number'].astype(str).str.strip()
        
        if 'total_gross_amount' in df.columns:
            df['total_gross_amount'] = pd.to_numeric(df['total_gross_amount'], errors='coerce').fillna(0)
        
        # --- REQUIREMENT B: AGGREGATION LOGIC ---
        agg_rules = {
            'total_gross_amount': 'sum',
            'invoice_date': 'first',
            'partner_name': 'first'
        }
        # Only include rules for columns that actually exist
        final_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        if not df.empty:
            print(f"Karton file rows before aggregation: {len(df)}")
            df = df.groupby('invoice_number').agg(final_agg_rules).reset_index()
            print(f"Karton file rows after aggregation: {len(df)}")

        standard_cols = ['invoice_number'] + list(final_agg_rules.keys())
        print("Success: Karton file loaded, standardized, and aggregated.")
        return df[standard_cols]
        
    except Exception as e:
        st.error(f"Error loading the Karton file: {e}")
        return pd.DataFrame()


def load_minta_file(uploaded_file):
    """
    Loads the 'mintavétel' Excel file, handles its complex header, 
    standardizes columns, and aggregates rows by invoice number.
    """
    print("Loading Minta file...")
    try:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, engine='calamine', skiprows=9, header=None) # Using calamine for better compatibility
        if df.empty or df.iloc[0].isnull().all():
            raise ValueError("The file appears to be empty or header is missing after skipping 9 rows.")

        df.columns = df.iloc[0]
        df = df.drop(df.index[0]).reset_index(drop=True)
        df.columns = [str(col).strip() if col is not None else f'unnamed_{i}' for i, col in enumerate(df.columns)]

        print("DataFrame after manually setting header:")
        print(df.head())
        
        COLUMN_MAP = {
            'Bizonylatszám': 'invoice_number',
            'Érték': 'total_net_amount',
            'Bizonylat dátuma': 'invoice_date',
            'Partner': 'partner_name'
        }
        df.rename(columns=COLUMN_MAP, inplace=True)
        
        if 'invoice_number' not in df.columns:
            raise KeyError("The required column 'Bizonylatszám' was not found.")

        # Standardize key column types and drop rows with no invoice number
        df.dropna(subset=['invoice_number'], inplace=True)
        df['invoice_number'] = df['invoice_number'].astype(str).str.strip()
        if 'total_gross_amount' in df.columns:
            df['total_gross_amount'] = pd.to_numeric(df['total_gross_amount'], errors='coerce').fillna(0)

        # --- REQUIREMENT B: AGGREGATION LOGIC ---
        agg_rules = {
            'total_gross_amount': 'sum',
            'invoice_date': 'first',
            'partner_name': 'first'
        }
        final_agg_rules = {k: v for k, v in agg_rules.items() if k in df.columns}

        if not df.empty:
            print(f"Minta file rows before aggregation: {len(df)}")
            df = df.groupby('invoice_number').agg(final_agg_rules).reset_index()
            print(f"Minta file rows after aggregation: {len(df)}")
            
        standard_cols = ['invoice_number'] + list(final_agg_rules.keys())
        print("Success: Minta file loaded, standardized, and aggregated.")
        return df[standard_cols]

    except Exception as e:
        st.error(f"Error processing the Minta file: {e}")
        return pd.DataFrame()

# --- REFINED COMPARISON FUNCTION ---

def perform_comparison(processed_df, comparison_df, source_name):
    """
    Compares the main processed_df against a standardized comparison_df.

    Args:
        processed_df (pd.DataFrame): The main dataframe with standard column names.
        comparison_df (pd.DataFrame): A dataframe from another source, with standardized column names.
        source_name (str): The name of the source (e.g., "NAV", "Karton").
    """
    if processed_df.empty or comparison_df.empty:
        st.warning(f"Cannot perform comparison for {source_name} due to an empty dataframe.")
        return pd.DataFrame()

    # Suffix for the comparison source columns
    comp_suffix = f"_{source_name.lower()}"
    
    # Add suffix to the comparison df before merging
    comparison_df_suffixed = comparison_df.add_suffix(comp_suffix)
    # Rename the key column back for a clean merge
    comparison_df_suffixed.rename(columns={f'invoice_number{comp_suffix}': 'invoice_number'}, inplace=True)

    # Perform a left merge to keep all records from the processed file
    merged_df = pd.merge(processed_df, comparison_df_suffixed, on='invoice_number', how='left')

    # --- Perform checks for each amount type ---
    
    # Check 1: Gross Amount
    gross_comp_col = f'total_gross_amount{comp_suffix}'
    if gross_comp_col in merged_df.columns:
        found_gross = merged_df[gross_comp_col].notna()
        gross_match = np.isclose(merged_df['total_gross_amount'], merged_df[gross_comp_col], equal_nan=True)
        merged_df[f'Bruttó egyezik? ({source_name})'] = np.where(found_gross, np.where(gross_match, '✅ IGAZ', '❌ HAMIS'), '➖ HIÁNYZIK')
    else:
        merged_df[f'Bruttó egyezik? ({source_name})'] = '➖ NEM TARTALMAZZA'

    # Check 2: Net Amount
    net_comp_col = f'total_net_amount{comp_suffix}'
    if net_comp_col in merged_df.columns:
        found_net = merged_df[net_comp_col].notna()
        net_match = np.isclose(merged_df['total_net_amount'], merged_df[net_comp_col], equal_nan=True)
        merged_df[f'Nettó egyezik? ({source_name})'] = np.where(found_net, np.where(net_match, '✅ IGAZ', '❌ HAMIS'), '➖ HIÁNYZIK')
    else:
        merged_df[f'Nettó egyezik? ({source_name})'] = '➖ NEM TARTALMAZZA'
        
    # Check 3: VAT Amount
    vat_comp_col = f'total_vat_amount{comp_suffix}'
    if vat_comp_col in merged_df.columns:
        found_vat = merged_df[vat_comp_col].notna()
        vat_match = np.isclose(merged_df['total_vat_amount'], merged_df[vat_comp_col], equal_nan=True)
        merged_df[f'ÁFA egyezik? ({source_name})'] = np.where(found_vat, np.where(vat_match, '✅ IGAZ', '❌ HAMIS'), '➖ HIÁNYZIK')
    else:
        merged_df[f'ÁFA egyezik? ({source_name})'] = '➖ NEM TARTALMAZZA'

    # Display the result columns next to the original and compared values
    display_cols = list(processed_df.columns)
    
    # Add comparison data columns that were found
    original_comparison_cols = [col for col in comparison_df.columns if col != 'invoice_number']
    for col in original_comparison_cols:
        suffixed_col = f"{col}{comp_suffix}"
        if suffixed_col in merged_df.columns:
            display_cols.append(suffixed_col)
            merged_df.rename(columns={suffixed_col: f"{col} ({source_name})"}, inplace=True)
            display_cols[-1] = f"{col} ({source_name})" # update the name in the display list
    
    # Add result columns
    result_cols = [
        f'Bruttó egyezik? ({source_name})', 
        f'Nettó egyezik? ({source_name})', 
        f'ÁFA egyezik? ({source_name})'
    ]
    display_cols.extend(result_cols)
    
    return merged_df[display_cols]

def generate_comparison_report(processed_file, nav_file, karton_file, minta_file):
    try:
        processed_df = load_processed_file(processed_file)
        nav_df = load_nav_file(nav_file)
        karton_df = load_karton_file(karton_file)
        minta_df = load_minta_file(minta_file)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None

    nav_report = perform_comparison(processed_df.copy(), nav_df, source_name="nav")
    karton_report = perform_comparison(processed_df.copy(), karton_df, source_name="karton")
    minta_report = perform_comparison(processed_df.copy(), minta_df, source_name="minta")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        nav_report.to_excel(writer, index=False, sheet_name='NAV Comparison')
        karton_report.to_excel(writer, index=False, sheet_name='Karton Comparison')
        minta_report.to_excel(writer, index=False, sheet_name='Minta Comparison')
    return output.getvalue()


# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="Batch Audit Assistant")

# Initialize session state
if "log" not in st.session_state: st.session_state.log = []
if "extracted_data" not in st.session_state: st.session_state.extracted_data = {}
if "output_excel_file" not in st.session_state: st.session_state.output_excel_file = None
if "auth_ok" not in st.session_state: st.session_state.auth_ok = initialize_vertex_ai()
if "comparison_result_bytes" not in st.session_state: st.session_state.comparison_result_bytes = None

st.title("Audit asszisztens")
st.markdown("PDF számlák adatainak kinyerése és külső nyilvántartásokkal való egyeztetése.")

tab1, tab2 = st.tabs(["1. Adatok kinyerése PDF számlákból", "2. Összevetés NAV, karton, minta Excel fájlokkal"])

with tab1:
    if not st.session_state.auth_ok:
        st.header("🔑 Hitelesítés szükséges")
        st.warning("A PDF feldolgozáshoz Google Cloud Service Account kulcs szükséges.")
        uploaded_key = st.file_uploader("Upload Service Account JSON Key", type=["json"])
        if uploaded_key:
            if save_credentials(uploaded_key):
                st.session_state.auth_ok = True
                st.success("Credentials saved! The app will now reload.")
                st.rerun()
    else:
        # st.success(f"✅ Authenticated to Google Cloud Project: **{config.GCP_PROJECT_ID}**")
        st.header("1. Lépés: PDF Számlák Feltöltése Feldolgozásra")
        btn_col1, btn_col2, btn_col3 = st.columns([2, 3, 2])
        with btn_col2:
            pdf_files = st.file_uploader("Válassza ki a PDF dokumentumokat", type=["pdf"], accept_multiple_files=True)
            if st.button("🚀 Kötegelt Feldolgozás Indítása", type="primary", use_container_width=True, disabled=not pdf_files):
                st.session_state.log = ["INFO: Kötegelt folyamat elindítva."]
                st.session_state.extracted_data = {}
                st.session_state.output_excel_file = None
                with st.spinner("Kötegelt feldolgozás fut... Részletek a naplóban."):
                    asyncio.run(run_batch_extraction(pdf_files))
                st.success("A kötegelt feldolgozás befejeződött!")

        if st.session_state.output_excel_file and os.path.exists(st.session_state.output_excel_file):
            st.subheader("Eredmények Letöltése")
            st.metric("Sikeresen Feldolgozott Számlák", f"{len(st.session_state.extracted_data)}")
            with open(st.session_state.output_excel_file, "rb") as file_data:
                # Centered download button
                dl_btn_col1, dl_btn_col2, dl_btn_col3 = st.columns([2, 3, 2])
                with dl_btn_col2:
                    st.download_button(
                        label="📥 Feldolgozott Adatok Letöltése (Excel)", data=file_data,
                        file_name=os.path.basename(st.session_state.output_excel_file),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                    )

with tab2:
    st.header("2. Lépés: Adatok Összehasonlítása és Jelentéskészítés")
    st.markdown("Töltse fel a feldolgozott Excel fájlt és a három összehasonlító fájlt a jelentés elkészítéséhez.")
    col1, col2 = st.columns(2)
    with col1:
        processed_excel = st.file_uploader("1. Feldolgozott Számlák Excel Feltöltése (az 1. lépésből)", type=["xlsx"])
        nav_excel = st.file_uploader("2. NAV Adat Excel Feltöltése", type=["xlsx", "xls"])
    with col2:
        karton_excel = st.file_uploader("3. Karton Adat Excel Feltöltése", type=["xlsx", "xls"])
        minta_excel = st.file_uploader("4. Minta Adat Excel Feltöltése", type=["xlsx", "xls"])

    # Centered button using columns
    comp_btn_col1, comp_btn_col2, comp_btn_col3 = st.columns([2, 3, 2])
    with comp_btn_col2:
        if st.button("🔍 Összehasonlítás és Jelentés Készítése", type="primary", use_container_width=True, disabled=not all([processed_excel, nav_excel, karton_excel, minta_excel])):
            with st.spinner("Jelentés készítése..."):
                report_bytes = generate_comparison_report(processed_excel, nav_excel, karton_excel, minta_excel)
                if report_bytes:
                    st.session_state.comparison_result_bytes = report_bytes
                    st.success("A jelentés sikeresen elkészült!")
                else:
                    st.error("A jelentés készítése sikertelen. Ellenőrizze a naplót.")

    if st.session_state.comparison_result_bytes:
        st.subheader("Összehasonlító Jelentés Letöltése")
        st.info("A jelentés három munkalapot tartalmaz: egyet-egyet a NAV, Karton és Minta összehasonlításokhoz.")
        # Centered download button
        rep_dl_btn_col1, rep_dl_btn_col2, rep_dl_btn_col3 = st.columns([2, 3, 2])
        with rep_dl_btn_col2:
            st.download_button(
                label="📥 Teljes Összehasonlító Jelentés Letöltése (Excel)",
                data=st.session_state.comparison_result_bytes,
                file_name=f"osszehasonlito_jelentes_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# --- Common Log Viewer ---
if st.session_state.log:
    with st.expander("📜 View Processing Log", expanded=False):
        log_text = "\n".join(reversed(st.session_state.log))
        st.text_area("Log", log_text, height=300, key="log_area", disabled=True)