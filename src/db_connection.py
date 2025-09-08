import json
import os
import psycopg2
import logging

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Use environment variables for sensitive data
# It's a best practice to store credentials in AWS Secrets Manager or Parameter Store
DB_HOST = "flexdataset.cluster-cpoeqq6cwu00.ap-southeast-2.rds.amazonaws.com"
DB_NAME = "FlexDataseterMaster"
DB_USER = "FlexUser"
DB_PASS = "Luffy123&&Lucky"
DB_PORT = "5432"

def get_connection():
    conn = None
    try:
        # Establish a connection to the database
        logger.info("Attempting to connect to the database...")
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            port=DB_PORT
        )
        logger.info("Successfully connected to the database.")
        cursor = conn.cursor()
        AbnCreateQuery = """
        CREATE TABLE IF NOT EXISTS abn (
            id SERIAL PRIMARY KEY,
            Abn VARCHAR(20),
            AbnStatus VARCHAR(50),
            AbnStatusEffectiveFrom TIMESTAMP NULL,
            Acn VARCHAR(20),
            AddressDate TIMESTAMP NULL,
            AddressPostcode VARCHAR(10),
            AddressState VARCHAR(10),
            BusinessName TEXT,
            EntityName TEXT,
            EntityTypeCode VARCHAR(20),
            EntityTypeName VARCHAR(100),
            Gst VARCHAR(5),
            Message TEXT,
            Contact TEXT,
            Website TEXT,
            Address TEXT,
            Email TEXT,
            SocialLink TEXT,
            Review TEXT,
            Industry TEXT,
            Documents TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        cursor.execute(AbnCreateQuery)
        return conn

    except Exception as e:
        logger.error(f"Error connecting to or querying the database: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }