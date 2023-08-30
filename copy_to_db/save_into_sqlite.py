import shutil
import os
import sqlite3
import pickle
import logging
import multiprocessing
import sys

# Configure the logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# def copy_empty_db_to_temp(data_path_suffix=""):
#     if not os.path.exists("/tmp"):
#        os.makedirs("/tmp")
#     source_file_db = f'/fast/rhe/copy_to_db/empty_database.db'
#     destination_folder_db = f'/tmp/database_{data_path_suffix}.db'
#     # destination_folder_db = f'/scratch/rhe/database_{data_path_suffix}.db'
#     shutil.copy(source_file_db, destination_folder_db)
#     logging.info(f"Copy empty database from {source_file_db} to {destination_folder_db}")
#

def copy_full_db_to_fast(data_path_suffix="", chunk=0):
    if not os.path.exists(f"/fast/rhe/results/mcrl/{data_path_suffix}"):
       os.makedirs(f"/fast/rhe/results/mcrl/{data_path_suffix}")
    source_file_db = f'/tmp/database_{data_path_suffix}_{chunk}.db'
    destination_folder_db = f"/fast/rhe/results/mcrl/{data_path_suffix}/database_{data_path_suffix}_{chunk}.db"
    shutil.copy(source_file_db, destination_folder_db)
    logging.info(f"Copy full database from {source_file_db} to {destination_folder_db}")

    # Delete the database from the temporary folder after copying it to the fast folder
    try:
        os.remove(source_file_db)
    except OSError:
        pass


def process_chunk(chunk_files, data_directory, data_path_suffix, chunk_index):
    logging.info(f"Starting processing chunk {chunk_index} for prior {data_path_suffix}")
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS data
                    (PID TEXT, MODEL TEXT, CONTENT BLOB,
                    PRIMARY KEY (PID, MODEL))''')
    conn.commit()

    cursor.execute('BEGIN TRANSACTION')

    for filename in chunk_files:
        try:
            PID, model_with_extension = filename.split('_')[0], filename.split('_')[1]
            model = os.path.splitext(model_with_extension)[0]

            with open(os.path.join(data_directory, filename), 'rb') as f:
                content = pickle.load(f)

            new_models = {model: content}
            new_models_data = pickle.dumps(new_models)

            cursor.execute("INSERT INTO data (PID, MODEL, CONTENT) VALUES (?, ?, ?)",
                           (PID, model, new_models_data))

        except Exception as e:
            logging.error(f"Error: {e}")
            logging.error(f"Error while processing {filename}")

    cursor.execute('COMMIT')
    conn.execute(f"vacuum main into '/tmp/database_{data_path_suffix}_{chunk_index}.db'")
    conn.close()

    # Copy the database to fast storage
    copy_full_db_to_fast(data_path_suffix=data_path_suffix, chunk=chunk_index)
    logging.info(f"Done with processing chunk {chunk_index} for prior {data_path_suffix}")


def main(data_path_suffix):
    data_directory = f"/work/rhe/mcl_toolbox/mcl_toolbox/results_correct_ones/mcrl/{data_path_suffix}"
    pickle_files = [filename for filename in os.listdir(data_directory)]
    logging.info(f"Starting processing of data for {data_path_suffix}")
    logging.info(f"Total number of pickle files: {len(pickle_files)}")

    chunk_size = 1000
    num_cores = multiprocessing.cpu_count()
    logging.info("Number of cores: %d", num_cores)

    chunks = [pickle_files[i:i + chunk_size] for i in range(0, len(pickle_files), chunk_size)]
    logging.info(f"Number of chunks for processing: {len(chunks)}")

    with multiprocessing.Pool(processes=num_cores) as pool:
        pool.starmap(process_chunk, [(chunk, data_directory, data_path_suffix, i) for i, chunk in enumerate(chunks)])


if __name__ == "__main__":
    data_path_suffix = sys.argv[1]
    main(data_path_suffix=data_path_suffix)
