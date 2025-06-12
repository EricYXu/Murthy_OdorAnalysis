import os
import zipfile
import shutil
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

bad_odors = []
# Returns a list of quantity values for the given file; blank values are replaced with -1
def extract_data_from_html(html_content, file_name):
    soup = BeautifulSoup(html_content, 'html.parser')
    div = soup.find('div', {'id': 'dA'})
    if not div:
        return [file_name, []]
    table = div.find('table')
    if not table:
        return [file_name, []]


    quantity_list = []
    compound_list = []


    for row in table.find_all('tr')[2:]:
        cells = row.find_all('td')
        if len(cells) > 3:
            quantity = cells[1].get_text(strip=True)
            compound = cells[2].get_text(strip=True) if cells[2].a else cells[3].get_text(strip=True)
            if not compound:
                if quantity != "":
                    bad_odors.append(file_name)
                    return None
                if quantity == "":
                    continue
            compound_list.append(compound)
            # If the quantity cell is blank, append -1, otherwise append the actual value
            if quantity == "":
                quantity_list.append(-1)
            else:
                quantity_list.append(quantity)


    return [file_name, quantity_list, compound_list]


# Function to recursively unzip all zip files in a directory
def unzip_all_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if zipfile.is_zipfile(file_path):
                print(f"Unzipping {file_path}...")
                extract_to = os.path.splitext(file_path)[0]
                os.makedirs(extract_to, exist_ok=True)
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                # After unzipping, check if there are more ZIPs inside
                unzip_all_files_in_directory(extract_to)  # Recursively unzip inner zips


# vcf_zip_path = "vcf/RawData.zip"
temp_dir = "temp"


# print(f"Unzipping {vcf_zip_path}...")
# with zipfile.ZipFile(vcf_zip_path, 'r') as zip_ref:
#     zip_ref.extractall(temp_dir)


# print("Recursively unzipping all inner zip files...")
# unzip_all_files_in_directory(temp_dir)


all_data = []


for root, dirs, files in os.walk(temp_dir):
    for file in files:
        if file.endswith(".html"):
            file_path = os.path.join(root, file)


            with open(file_path, 'r', encoding='utf-8') as html_file:
                html_content = html_file.read()


            data = extract_data_from_html(html_content, file)

            if data: all_data.append(data)


if all_data:
    df = pd.DataFrame(all_data, columns=['File Name', 'Quantity List', 'Compound List'])


    output_csv = "concentrations.csv"
    df.to_csv(output_csv, index=False)
    # convert bad_odors to csv
    bad_odors_df = pd.DataFrame(set(bad_odors), columns=['Odor Name'])
    bad_odors_df.to_csv('bad_odors.csv', index=False)
    print(f"Data extraction completed. The file is saved at: {output_csv}")
else:
    print("No data extracted from the HTML files.")
