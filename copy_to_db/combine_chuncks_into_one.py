import sqlite3
import os
import sys
import logging
import shutil

# Configure the logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def copy_from_fast_to_temp(data_path_suffix="", database_name=""):
    # Copy the database from fast to temp to read the data
    source_file_db = f'/fast/rhe/results/mcrl/{data_path_suffix}/{database_name}'
    destination_folder_db = f'/tmp/{database_name}'
    shutil.copy(source_file_db, destination_folder_db)
    logging.info(f"Copy database from {source_file_db} to {destination_folder_db}")

def copy_full_db_to_fast(data_path_suffix=""):
    # Copy finished large database from temp to fast
    source_file_db = f'/tmp/{data_path_suffix}.db'
    destination_folder_db = f"/fast/rhe/results/mcrl/{data_path_suffix}/{data_path_suffix}.db"
    shutil.copy(source_file_db, destination_folder_db)
    logging.info(f"Copy full database from {source_file_db} to {destination_folder_db}")

    # Delete the database from the temporary folder after copying it to the fast folder
    try:
        os.remove(source_file_db)
    except OSError:
        pass


def main(data_path_suffix=""):
    logging.info(f"Starting with processing databases for {data_path_suffix}")

    # Path to the folder containing your SQLite databases
    databases_folder = f"/fast/rhe/results/mcrl/{data_path_suffix}"

    # Name of the combined database
    # combined_db_name = f"/tmp/{data_path_suffix}.db"

    # Create a new combined database or connect to an existing one
    # combined_conn = sqlite3.connect(combined_db_name)
    combined_conn = sqlite3.connect(":memory:")
    combined_cursor = combined_conn.cursor()

    # Create a new table in the combined database to insert the data into
    combined_cursor.execute('''CREATE TABLE IF NOT EXISTS data
                        (PID TEXT, MODEL TEXT, CONTENT BLOB,
                        PRIMARY KEY (PID, MODEL))''')
    combined_conn.commit()

    # Iterate through each database in the folder
    for db_file in os.listdir(databases_folder):
        if db_file.endswith(".db"):
            logging.info(f"Processing file {db_file}")
            # db_path = os.path.join(databases_folder, db_file)

            # Copy the database from fast to temp
            copy_from_fast_to_temp(data_path_suffix=data_path_suffix, database_name=db_file)

            # Connect to each individual database
            individual_conn = sqlite3.connect(f'/tmp/{db_file}')
            individual_cursor = individual_conn.cursor()

            # Get the table name from the database file name (excluding the ".db" extension)
            table_name = "data"

            # Fetch all rows from the individual table
            individual_cursor.execute(f"SELECT * FROM {table_name}")
            rows = individual_cursor.fetchall()

            # Insert fetched rows into the combined database using executemany
            combined_cursor.executemany(f"INSERT INTO {table_name} VALUES ({','.join(['?'] * len(rows[0]))})", rows)

            # Commit changes for the individual database and close the connection
            individual_conn.commit()
            individual_conn.close()

            # Delete the database from the temporary folder after is has been processed
            try:
                os.remove(f'/tmp/{db_file}')
            except OSError:
                pass

            logging.info(f"Finished with file {db_file}")

    # Commit changes for the combined database and close the connection
    combined_conn.commit()
    combined_conn.execute(f"vacuum main into '/tmp/{data_path_suffix}.db'")
    combined_conn.close()

    copy_full_db_to_fast(data_path_suffix=data_path_suffix)

    logging.info(f"Finished with processing databases for {data_path_suffix}")


if __name__ == "__main__":
    data_path_suffix = sys.argv[1]
    main(data_path_suffix=data_path_suffix)