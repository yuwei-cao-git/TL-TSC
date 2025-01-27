import pandas as pd
import requests

# Path to your CSV file
csv_file_path = './FRI_Tile_Index_OVF.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    tile_name = row['Tilename']
    download_link = row['Download_H']

    # Download the file from the link
    response = requests.get(download_link)
    if response.status_code == 200:
        # Save the file to the disk
        with open(f'data/{tile_name}.laz', 'wb') as file:
            file.write(response.content)
        print(f'Successfully downloaded data/{tile_name}.laz')
    else:
        print(f'Failed to download data/{tile_name}.laz')