import json
import re
import time
import os
import logging
import concurrent.futures
from typing import Any, Dict, List, Tuple

import requests
import pandas as pd
import psycopg2
import psycopg2.extras
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google import genai
from google.genai import types
from pydantic import BaseModel
from .db_connection import get_connection

# --- DB insert function ---
def insert_batch_to_postgres(batch_df: pd.DataFrame) -> None:
    if batch_df.empty:
        return

    conn = get_connection()
    if not isinstance(conn, psycopg2.extensions.connection):
        # Defensive: get_connection might have returned an error dict
        raise RuntimeError("Database connection is not available")

    cursor = conn.cursor()

    # Ensure only known columns
    header_columns = [
        'Abn', 'AbnStatus', 'AbnStatusEffectiveFrom', 'Acn', 'AddressDate',
        'AddressPostcode', 'AddressState', 'BusinessName', 'EntityName',
        'EntityTypeCode', 'EntityTypeName', 'Gst', 'Message', 'Contact',
        'Website', 'Address', 'Email', 'SocialLink', 'Review', 'Industry',
        'Documents'
    ]
    # Add any missing expected columns with empty default values
    for col in header_columns:
        if col not in batch_df.columns:
            batch_df[col] = ""
    batch_df = batch_df[header_columns]

    # Convert lists/dicts to JSON strings for text columns
    batch_df['SocialLink'] = batch_df['SocialLink'].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else (x if x is not None else "")
    )
    batch_df['Documents'] = batch_df['Documents'].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else (x if x is not None else "")
    )

    values = [tuple(x) for x in batch_df.to_numpy()]

    insert_query = """
        INSERT INTO abn
        (Abn, AbnStatus, AbnStatusEffectiveFrom, Acn, AddressDate, AddressPostcode,
         AddressState, BusinessName, EntityName, EntityTypeCode, EntityTypeName, Gst,
         Message, Contact, Website, Address, Email, SocialLink, Review, Industry, Documents)
        VALUES %s
    """

    psycopg2.extras.execute_values(
        cursor, insert_query, values, template=None, page_size=1000
    )

    conn.commit()
    cursor.close()
    # Intentionally do not close the connection to enable reuse across invocations
# --- DB insert function end ---


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Configuration via environment variables ---
ABR_AUTH_GUID = os.getenv("ABR_AUTH_GUID", "250e9f55-f46e-4104-b0df-774fa28cff97")
GENAI_API_KEY = os.getenv("GENAI_API_KEY", "AIzaSyD1VmH7wuQVqxld5LeKjF79eRq1gqVrNFA")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "200"))
GENAI_BATCH_SIZE = int(os.getenv("GENAI_BATCH_SIZE", "50"))
ABN_DETAILS_CONCURRENCY = int(os.getenv("ABN_DETAILS_CONCURRENCY", "5"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "10"))
BATCH_PAUSE_SECONDS = float(os.getenv("BATCH_PAUSE_SECONDS", "0"))
GST_FILTER = os.getenv("GST_FILTER", "")  # "y", "n", or empty for both
MAX_POSTCODES = int(os.getenv("MAX_POSTCODES", "0"))
POSTCODE_START_INDEX = int(os.getenv("POSTCODE_START_INDEX", "0"))

# --- HTTP session with retries ---
def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=50, pool_maxsize=50)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

HTTP_SESSION = _build_session()

# --- Initialize GenAI client once per container ---
GENAI_CLIENT = genai.Client(api_key=GENAI_API_KEY) if GENAI_API_KEY else None


# --- Models (moved to module scope to avoid redefinition) ---
class ABNDetails(BaseModel):
    Contact: str
    Website: str
    Address: str
    Email: str
    SocialLink: List[str]
    review: str
    Industry: str


class ACNDocumentsDetails(BaseModel):
    documentid: str
    dateofpublication: str
    noticetype: str


# --- Helpers for robust ACN Docs parsing ---
def _find_json_array_span(text: str) -> tuple[int, int] | None:
    start_index = text.find("[")
    if start_index == -1:
        return None
    depth = 0
    for index in range(start_index, len(text)):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return start_index, index + 1
    return None


def _extract_json_array(text: str) -> list:
    if not text:
        return []
    span = _find_json_array_span(text)
    if not span:
        # Try to quickly sanitize backticks or code fences
        sanitized = text.strip().strip("`")
        try:
            parsed = json.loads(sanitized)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    start_index, end_index = span
    snippet = text[start_index:end_index]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _coerce_acn_docs(items: list) -> list[ACNDocumentsDetails]:
    coerced: list[ACNDocumentsDetails] = []
    for raw in items[:3]:
        if not isinstance(raw, dict):
            continue
        document_id = str(raw.get("documentid", ""))[:80]
        publication_date = str(raw.get("dateofpublication", ""))[:80]
        notice_type = str(raw.get("noticetype", ""))[:80]
        coerced.append(
            ACNDocumentsDetails(
                documentid=document_id,
                dateofpublication=publication_date,
                noticetype=notice_type,
            )
        )
    return coerced


def _generate_acn_docs_with_fallback(batch_prompts: list[str]) -> list[ACNDocumentsDetails]:
    results: list[ACNDocumentsDetails] = []
    for prompt in batch_prompts:
        try:
            resp = GENAI_CLIENT.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt],
                config={
                    "response_mime_type": "text/plain",
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                },
            )
            raw_text = getattr(resp, "text", None)
            if not raw_text and hasattr(resp, "candidates") and resp.candidates:
                # Fallback extraction for SDK variations
                parts = getattr(resp.candidates[0], "content", None)
                raw_text = getattr(parts, "parts", [None])[0]
                raw_text = getattr(raw_text, "text", None)
            items = _extract_json_array(raw_text or "")
            coerced = _coerce_acn_docs(items)
            if coerced:
                results.append(coerced[0])
            else:
                results.append(ACNDocumentsDetails(documentid="", dateofpublication="", noticetype=""))
        except Exception:
            results.append(ACNDocumentsDetails(documentid="", dateofpublication="", noticetype=""))
    return results


# --- Load postcodes once (if available) ---
def _load_postcodes() -> List[str]:
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.normpath(os.path.join(base_dir, "..", "data", "AustralianPostcodesUnique.csv"))
        postcodes_df = pd.read_csv(csv_path)
        return postcodes_df['postcode'].astype(str).tolist()
    except Exception as e:
        logger.warning(f"Could not load postcodes CSV: {e}")
        return []


POSTCODES_CACHE = _load_postcodes()


def _search_abns(postcode: str, gst: str) -> List[str]:
    url = "https://abr.business.gov.au/ABRXMLSearch/AbrXmlSearch.asmx/SearchByABNStatus"
    params = {
        "postcode": str(postcode),
        "activeABNsOnly": "y",
        "currentGSTRegistrationOnly": gst,
        "entityTypeCode": "",
        "authenticationGuid": ABR_AUTH_GUID,
    }
    response = HTTP_SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    ns = {'ns': 'http://abr.business.gov.au/ABRXMLSearch/'}
    abns = [abn.text for abn in root.findall('.//ns:abn', ns) if abn is not None and abn.text]
    return abns


def _fetch_abn_details(abn: str) -> Dict[str, Any]:
    abn_clean = (abn or "").replace(" ", "")
    url = f"https://abr.business.gov.au/json/AbnDetails.aspx"
    params = {"abn": abn_clean, "callback": "callback", "guid": ABR_AUTH_GUID}
    resp = HTTP_SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.text.replace("callback(", "").rstrip(")")
    payload = json.loads(data)
    return payload


def lambda_handler(event, context):
    if not ABR_AUTH_GUID:
        logger.error("ABR_AUTH_GUID is not configured. Set it via environment variable.")
        return {"statusCode": 500, "body": "Missing ABR_AUTH_GUID"}

    postcodes_list = POSTCODES_CACHE[:]
    if POSTCODE_START_INDEX > 0:
        postcodes_list = postcodes_list[POSTCODE_START_INDEX:]
    if MAX_POSTCODES > 0:
        postcodes_list = postcodes_list[:MAX_POSTCODES]

    gst_values = [GST_FILTER] if GST_FILTER in ("y", "n") else ["y", "n"]

    # --- MAIN LOOP ---
    for gst_param in gst_values:
        for postcode in postcodes_list:
            try:
                abns = _search_abns(postcode, gst_param)
                logger.info(f"Found {len(abns)} ABNs for postcode {postcode} with GST {gst_param}")

                # De-duplicate while preserving order
                seen = set()
                abns = [a for a in abns if not (a in seen or seen.add(a))]

                for i in range(0, len(abns), BATCH_SIZE):
                    abn_batch = abns[i:i + BATCH_SIZE]
                    batch_data: List[Dict[str, Any]] = []

                    # --- Fetch ABN details concurrently with bounded concurrency ---
                    def _safe_fetch(a: str) -> Tuple[str, Dict[str, Any] | None]:
                        try:
                            details = _fetch_abn_details(a)
                            return a, details
                        except requests.RequestException as e:
                            logger.warning(f"Error fetching ABN {a}: {e}")
                            return a, None

                    with concurrent.futures.ThreadPoolExecutor(max_workers=ABN_DETAILS_CONCURRENCY) as pool:
                        results = list(pool.map(_safe_fetch, abn_batch))

                    for abn_value, details in results:
                        if details is None:
                            continue
                        batch_data.append(details)

                    if not batch_data:
                        logger.info(f"No ABN details fetched for batch {i // BATCH_SIZE + 1} (postcode {postcode}, GST {gst_param})")
                        continue

                    # --- Build prompts for GenAI where ACN exists ---
                    genai_prompts: List[str] = []
                    genai_indices: List[int] = []
                    acn_genai_prompts: List[str] = []
                    acn_genai_indices: List[int] = []

                    for idx, item in enumerate(batch_data):
                        acn_value = item.get('Acn', "") or ""
                        if acn_value:
                            entity_name = item.get('EntityName') or (item.get('BusinessName')[0] if item.get('BusinessName') else "")
                            state_code = item.get('AddressState', "")
                            genai_prompts.append(
                                f"give me the website, contact number, social media links, total reviews, Industry and address of '{entity_name}', {state_code}, Australia. I want review in format of 4/5 like that"
                            )
                            genai_indices.append(idx)
                            acn_genai_prompts.append(
                                (
                                    "Using only the site https://publishednotices.asic.gov.au/browsesearch-notices, "
                                    f"return a JSON array with at most 3 objects listing notices for ACN {acn_value}. "
                                    "Each object must have exactly these keys: documentid, dateofpublication, noticetype. "
                                    "Respond with JSON only, no prose or markdown. Keep values concise (<=80 characters)."
                                )
                            )
                            acn_genai_indices.append(idx)

                    # --- Call Generative AI in batches ---
                    genai_results: List[ABNDetails] = []
                    if GENAI_CLIENT and genai_prompts:
                        for j in range(0, len(genai_prompts), GENAI_BATCH_SIZE):
                            batch_prompts = genai_prompts[j:j + GENAI_BATCH_SIZE]
                            try:
                                response = GENAI_CLIENT.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=batch_prompts,
                                    config={
                                        "response_mime_type": "application/json",
                                        "response_schema": list[ABNDetails],
                                    },
                                )
                                genai_results.extend(response.parsed)
                            except Exception as e:
                                logger.warning(f"Error in Generative AI batch call (ABN): {e}")
                                genai_results.extend([
                                    ABNDetails(Contact="", Website="", Address="", Email="", SocialLink=[], review="", Industry="")
                                    for _ in batch_prompts
                                ])
                    else:
                        # Fill with empty results if client not configured
                        genai_results = [ABNDetails(Contact="", Website="", Address="", Email="", SocialLink=[], review="", Industry="") for _ in genai_prompts]

                    acn_genai_results: List[ACNDocumentsDetails] = []
                    if GENAI_CLIENT and acn_genai_prompts:
                        for j in range(0, len(acn_genai_prompts), GENAI_BATCH_SIZE):
                            batch_acn_prompts = acn_genai_prompts[j:j + GENAI_BATCH_SIZE]
                            try:
                                dresponse = GENAI_CLIENT.models.generate_content(
                                    model="gemini-2.5-flash",
                                    contents=batch_acn_prompts,
                                    config={
                                        "response_mime_type": "application/json",
                                        "response_schema": list[ACNDocumentsDetails],
                                        "temperature": 0.2,
                                        "max_output_tokens": 1024,
                                    },
                                )
                                acn_genai_results.extend(dresponse.parsed)
                            except Exception as e:
                                logger.warning(f"Error in Generative AI batch call (ACN Docs): {e}")
                                # Per-prompt fallback parsing
                                fallback_results = _generate_acn_docs_with_fallback(batch_acn_prompts)
                                acn_genai_results.extend(fallback_results)
                    else:
                        acn_genai_results = [ACNDocumentsDetails(documentid="", dateofpublication="", noticetype="") for _ in acn_genai_prompts]

                    # --- Combine results back into batch_data ---
                    for offset, idx in enumerate(genai_indices):
                        if offset < len(genai_results):
                            r = genai_results[offset]
                            batch_data[idx]['Contact'] = r.Contact
                            batch_data[idx]['Website'] = r.Website
                            batch_data[idx]['Address'] = r.Address
                            batch_data[idx]['Email'] = r.Email
                            batch_data[idx]['SocialLink'] = r.SocialLink
                            batch_data[idx]['Review'] = r.review
                            batch_data[idx]['Industry'] = r.Industry

                    for offset, idx in enumerate(acn_genai_indices):
                        if offset < len(acn_genai_results):
                            doc_item = acn_genai_results[offset]
                            batch_data[idx]['Documents'] = [f"{doc_item.documentid},{doc_item.dateofpublication},{doc_item.noticetype}"]

                    # Ensure defaults for items without ACN or missing fields
                    processed_batch_data: List[Dict[str, Any]] = []
                    for item in batch_data:
                        item.setdefault('Contact', "")
                        item.setdefault('Website', "")
                        item.setdefault('Address', "")
                        item.setdefault('Email', "")
                        item.setdefault('SocialLink', [])
                        item.setdefault('Review', "")
                        item.setdefault('Industry', "")
                        item.setdefault('Documents', [])
                        processed_batch_data.append(item)

                    # --- Insert into DB ---
                    batch_df = pd.json_normalize(processed_batch_data)
                    if not batch_df.empty:
                        insert_batch_to_postgres(batch_df)
                        logger.info(f"Inserted batch {i // BATCH_SIZE + 1} for postcode {postcode} with GST {gst_param}")
                    else:
                        logger.info(f"No data in batch {i // BATCH_SIZE + 1} for postcode {postcode} with GST {gst_param}")

                    if BATCH_PAUSE_SECONDS > 0:
                        time.sleep(BATCH_PAUSE_SECONDS)

            except requests.exceptions.RequestException as e:
                logger.warning(f"Error fetching ABNs for postcode {postcode} with GST {gst_param}: {e}")

def _event_from_env():
    payload = os.getenv("FC_EVENT_JSON")
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except Exception:
        return {}

def main():
    event = _event_from_env()
    lambda_handler(event, None)

if __name__ == "__main__":
    main()
