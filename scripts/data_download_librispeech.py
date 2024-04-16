import requests
import pandas as pd
import soundfile as sf
import os

def download_parquet_file(url, file_path):
    """
    Downloads a Parquet file from a given URL and saves it to a specified path.

    Parameters:
    - url (str): The URL from where to download the file.
    - file_path (str): The local path where the file will be saved.

    Returns:
    - None
    """
    response = requests.get(url)
    response.raise_for_status()  # will raise an exception for HTTP error codes

    with open(file_path, 'wb') as f:
        f.write(response.content)
    print("Downloaded the Parquet file successfully.")

def load_dataframe(file_path):
    """
    Loads a DataFrame from a Parquet file.

    Parameters:
    - file_path (str): The path of the Parquet file.

    Returns:
    - df (DataFrame): A pandas DataFrame loaded from the Parquet file.
    """
    return pd.read_parquet(file_path)

def test_flac_data(byte_data, test_file_path):
    """
    Writes byte data to a file and prints a confirmation message.

    Parameters:
    - byte_data (bytes): Audio data to be written to file.
    - test_file_path (str): Path where the audio data will be saved.

    Returns:
    - None
    """
    with open(test_file_path, 'wb') as f:
        f.write(byte_data)
    print(f"Data written to {test_file_path}")

def convert_flac_to_wav(file_paths):
    """
    Converts a list of FLAC files to WAV format.

    Parameters:
    - file_paths (list): A list of paths to FLAC files.

    Returns:
    - None
    """
    for file_path in file_paths:
        output_file = file_path.replace('.flac', '.wav')
        data, samplerate = sf.read(file_path)
        sf.write(output_file, data, samplerate)
        print(f"Converted {file_path} to WAV and saved to {output_file}")

def main():
    """
    Main function to orchestrate the downloading, loading, processing, and converting of audio files.

    Returns:
    - None
    """
    url = "https://huggingface.co/api/datasets/librispeech_asr/parquet/all/train.clean.100/0.parquet"
    file_path = "/Users/sukriti/Desktop/838c-project/train.clean.100_0.parquet"
    download_parquet_file(url, file_path)

    df = load_dataframe(file_path)
    random_rows = df.sample(n=5)
    file_dir = '/Users/sukriti/Desktop/838c-project'
    file_paths = []

    for index, row in random_rows.iterrows():
        test_file_path = os.path.join(file_dir, row['file'])
        file_paths.append(test_file_path)
        audio_bytes = row['audio']['bytes']
        test_flac_data(audio_bytes, test_file_path)

    convert_flac_to_wav(file_paths)

if __name__ == "__main__":
    main()
