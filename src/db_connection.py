import json
import os
import psycopg2
import logging

# Set up logging
logger = logging.getLogger(__name__)
conn = None

# Use environment variables for sensitive data
# It's a best practice to store credentials in AWS Secrets Manager or Parameter Store
DB_HOST = "flexdataset.cluster-cpoeqq6cwu00.ap-southeast-2.rds.amazonaws.com"
DB_NAME = "FlexDataseterMaster"
DB_USER = "FlexUser"
DB_PASS = "Luffy123&&Lucky"
DB_PORT = "5432"

def get_connection():
    global conn
    try:
        # Establish a connection to the database
        if conn is None or conn.closed: 
            logger.info("Attempting to connect to the database...")
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASS,
                port=DB_PORT
            )
            logger.info("Successfully connected to the database.")
            conn.autocommit = False 
              
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
            Gst TIMESTAMP NULL,
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

        # Remove duplicate ABN rows keeping the lowest id to allow unique index creation
        DedupQuery = """
        WITH ranked AS (
            SELECT id, Abn,
                   ROW_NUMBER() OVER (PARTITION BY Abn ORDER BY id) AS rn
            FROM abn
            WHERE Abn IS NOT NULL AND Abn <> ''
        )
        DELETE FROM abn a
        USING ranked r
        WHERE a.id = r.id AND r.rn > 1;
        """
        cursor.execute(DedupQuery)

        # Ensure unique constraint on Abn for idempotent upserts
        UniqueIdxQuery = """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname = 'abn_unique_abn' AND n.nspname = 'public'
            ) THEN
                CREATE UNIQUE INDEX abn_unique_abn ON abn (Abn);
            END IF;
        END$$;
        """
        cursor.execute(UniqueIdxQuery)

        # Create a simple key-value table for progress tracking
        KVCreateQuery = """
        CREATE TABLE IF NOT EXISTS kv_store (
            k TEXT PRIMARY KEY,
            v TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        cursor.execute(KVCreateQuery)
        return conn

    except Exception as e:
        logger.error(f"Error connecting to or querying the database: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error: {str(e)}')
        }


def get_kv_value(key: str):
    """Return value for key from kv_store, or None if missing. Uses existing connection."""
    database_connection = get_connection()
    if not isinstance(database_connection, psycopg2.extensions.connection):
        return None
    cursor = database_connection.cursor()
    cursor.execute("SELECT v FROM kv_store WHERE k = %s", (key,))
    row = cursor.fetchone()
    cursor.close()
    return row[0] if row else None


def set_kv_value(key: str, value: str) -> None:
    """Upsert key/value into kv_store. Commits immediately."""
    database_connection = get_connection()
    if not isinstance(database_connection, psycopg2.extensions.connection):
        raise RuntimeError("Database connection is not available")
    cursor = database_connection.cursor()
    cursor.execute(
        """
        INSERT INTO kv_store (k, v)
        VALUES (%s, %s)
        ON CONFLICT (k) DO UPDATE SET v = EXCLUDED.v, updated_at = NOW()
        """,
        (key, value),
    )
    database_connection.commit()
    cursor.close()


def get_last_processed_postcode(gst_flag: str):
    key = f"last_postcode_{gst_flag}"
    return get_kv_value(key)


def set_last_processed_postcode(gst_flag: str, postcode: str) -> None:
    key = f"last_postcode_{gst_flag}"
    set_kv_value(key, postcode)