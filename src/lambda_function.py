import json
import time
import os
import requests
import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras
import xml.etree.ElementTree as ET
from google import genai
from google.genai import types
from pydantic import BaseModel
from .db_connection import get_connection

# --- DB insert function ---
def insert_batch_to_postgres(batch_df):
    if batch_df.empty:
        return

    conn = get_connection()
    cursor = conn.cursor()

    # Ensure only known columns
    header_columns = [
        'Abn', 'AbnStatus', 'AbnStatusEffectiveFrom', 'Acn', 'AddressDate',
        'AddressPostcode', 'AddressState', 'BusinessName', 'EntityName',
        'EntityTypeCode', 'EntityTypeName', 'Gst', 'Message', 'Contact',
        'Website', 'Address', 'Email', 'SocialLink', 'Review', 'Industry',
        'Documents'
    ]
    batch_df = batch_df[header_columns]

    # Convert lists/dicts to JSON strings
    batch_df['SocialLink'] = batch_df['SocialLink'].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
    )
    batch_df['Documents'] = batch_df['Documents'].apply(
        lambda x: json.dumps(x) if isinstance(x, (list, dict)) else str(x)
    )

    values = [tuple(x) for x in batch_df.to_numpy()]

    insert_query = """
        INSERT INTO business_data
        (Abn, AbnStatus, AbnStatusEffectiveFrom, Acn, AddressDate, AddressPostcode,
         AddressState, BusinessName, EntityName, EntityTypeCode, EntityTypeName, Gst,
         Message, Contact, Website, Address, Email, SocialLink, Review, Industry, Documents)
        VALUES %s
    """

    psycopg2.extras.execute_values(
        cursor, insert_query, values, template=None, page_size=100
    )

    conn.commit()
    cursor.close()
    conn.close()
# --- DB insert function end ---


def lambda_handler(event, context):
    file_path = os.getenv("POSTCODE_SOURCE", "/var/task/data/uniqpostcode.csv")
    postcodes_df = pd.read_csv(file_path)
    postcodes_list = postcodes_df['postcode'].tolist()

    ABRAuthGuid = "250e9f55-f46e-4104-b0df-774fa28cff97"
    GenAiAPI = "AIzaSyD1VmH7wuQVqxld5LeKjF79eRq1gqVrNFA"

    class ABNDetails(BaseModel):
        Contact: str
        Website: str
        Address: str
        Email: str
        SocialLink: list[str]
        review: str
        Industry: str

    class ACNDocumentsDetails(BaseModel):
        documentid: str
        dateofpublication: str
        noticetype: str


    GstRegistrationValues = ['y', 'n']
    batch_size = 5
    genai_batch_size = 5

    # --- MAIN LOOP ---
    for GstParam in GstRegistrationValues:
        for postcode in postcodes_list:
            URL = "https://abr.business.gov.au/ABRXMLSearch/AbrXmlSearch.asmx/SearchByABNStatus"
            params = {
                "postcode": str(postcode),
                "activeABNsOnly": "y",
                "currentGSTRegistrationOnly": GstParam,
                "entityTypeCode": "",
                "authenticationGuid": ABRAuthGuid
            }

            try:
                response = requests.get(URL, params=params)
                response.raise_for_status()

                root = ET.fromstring(response.text)
                ns = {'ns': 'http://abr.business.gov.au/ABRXMLSearch/'}
                abns = [abn.text for abn in root.findall('.//ns:abn', ns)]
                print(f"Found {len(abns)} ABNs for postcode {postcode} with GST {GstParam}")

                for i in range(0, len(abns), batch_size):
                    abn_batch = abns[i:i + batch_size]
                    batch_data, genai_prompts, acn_genai_prompts = [], [], []
                    genai_results, acn_genai_results, processed_batch_data = [], [], []

                    for abn in abn_batch:
                        ABNClean = abn.replace(" ", "")
                        ABNDetailsAPI = f"https://abr.business.gov.au/json/AbnDetails.aspx?abn={ABNClean}&callback=callback&guid={ABRAuthGuid}"

                        try:
                            response = requests.get(ABNDetailsAPI)
                            response.raise_for_status()
                            data = response.text.replace("callback(", "").rstrip(")")
                            ResultABNDetails = json.loads(data)
                            batch_data.append(ResultABNDetails)

                            if ResultABNDetails['Acn'] != "":
                                CNameClean = ResultABNDetails['EntityName'] if not ResultABNDetails['BusinessName'] else ResultABNDetails['BusinessName'][0]
                                CState = ResultABNDetails['AddressState']
                                genai_prompts.append(
                                    f"give me the website, contact number, social media links, total reviews, Industry and address of '{CNameClean}', {CState}, Australia. I want review in format of 4/5 like that"
                                )
                                acn_genai_prompts.append(
                                    "Also using this site https://publishednotices.asic.gov.au/browsesearch-notices fetch all the notices related to ACN " + ResultABNDetails['Acn']
                                )

                            time.sleep(10)

                        except requests.exceptions.RequestException as e:
                            print(f"Error fetching ABN {ABNClean}: {e}")

                    # --- Generative AI ---
                    for j in range(0, len(genai_prompts), genai_batch_size):
                        batch_prompts = genai_prompts[j:j + genai_batch_size]
                        try:
                            response = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=batch_prompts,
                                config={
                                    "response_mime_type": "application/json",
                                    "response_schema": list[ABNDetails],
                                },
                            )
                            genai_results.extend(response.parsed)
                        except Exception as e:
                            print(f"Error in Generative AI batch call (ABN): {e}")
                            genai_results.extend([
                                ABNDetails(Contact="", Website="", Address="", Email="", SocialLink=[], review="", Industry="") for _ in batch_prompts
                            ])

                    for j in range(0, len(acn_genai_prompts), genai_batch_size):
                        batch_acn_prompts = acn_genai_prompts[j:j + genai_batch_size]
                        try:
                            Dresponse = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=batch_acn_prompts,
                                config={
                                    "response_mime_type": "application/json",
                                    "response_schema": list[ACNDocumentsDetails],
                                },
                            )
                            acn_genai_results.extend(Dresponse.parsed)
                        except Exception as e:
                            print(f"Error in Generative AI batch call (ACN Docs): {e}")
                            acn_genai_results.extend([
                                ACNDocumentsDetails(documentid="", dateofpublication="", noticetype="") for _ in batch_acn_prompts
                            ])

                    # --- Combine ---
                    genai_index = 0
                    acn_genai_index = 0
                    for item in batch_data:
                        if item['Acn'] != "" and genai_index < len(genai_results) and acn_genai_index < len(acn_genai_results):
                            item['Contact'] = genai_results[genai_index].Contact
                            item['Website'] = genai_results[genai_index].Website
                            item['Address'] = genai_results[genai_index].Address
                            item['Email'] = genai_results[genai_index].Email
                            item['SocialLink'] = genai_results[genai_index].SocialLink
                            item['Review'] = genai_results[genai_index].review
                            item['Industry'] = genai_results[genai_index].Industry

                            ResultABNDocuments = []
                            if acn_genai_index < len(acn_genai_results):
                                doc_item = acn_genai_results[acn_genai_index]
                                ResultABNDocuments.append(f"{doc_item.documentid},{doc_item.dateofpublication},{doc_item.noticetype}")
                                acn_genai_index += 1
                            item['Documents'] = ResultABNDocuments
                            genai_index += 1
                        else:
                            item['Contact'] = ""
                            item['Website'] = ""
                            item['Address'] = ""
                            item['Email'] = ""
                            item['SocialLink'] = ""
                            item['Review'] = ""
                            item['Industry'] = ""
                            item['Documents'] = ""

                        processed_batch_data.append(item)

                    # --- Insert into DB ---
                    batch_df = pd.json_normalize(processed_batch_data)
                    if not batch_df.empty:
                        insert_batch_to_postgres(batch_df)
                        print(f"Inserted batch {i//batch_size + 1} for postcode {postcode} with GST {GstParam}")
                    else:
                        print(f"No data in batch {i//batch_size + 1} for postcode {postcode} with GST {GstParam}")

                    time.sleep(60)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching ABNs for postcode {postcode} with GST {GstParam}: {e}")
