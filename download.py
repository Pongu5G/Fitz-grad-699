import os
import pandas as pd
import urllib.request
from tqdm import tqdm

# === ✅ CONFIG ===
MASTER_FOLDER = "fitz_master"                # All 17k images will go here
WORKING_FOLDER = "fitz_working_copy"         # Store CSVs here
ANNOTATIONS_FILE = "fitzpatrick17k.csv"      # Full dataset file
FULL_CSV_OUT = "fitz_full.csv"               # Saved copy of all rows

# === ✅ Create folders ===
os.makedirs(MASTER_FOLDER, exist_ok=True)
os.makedirs(WORKING_FOLDER, exist_ok=True)

# === ✅ Load CSV
df = pd.read_csv(ANNOTATIONS_FILE)
print(f"🔢 Loaded annotations: {len(df)} rows")

# === ✅ Use ALL rows (no filtering)
subset_df = df.copy()

# === ✅ Save full CSV to working folder
full_csv_path = os.path.join(WORKING_FOLDER, FULL_CSV_OUT)
subset_df.to_csv(full_csv_path, index=False)
print(f"📄 Saved full CSV to: {full_csv_path}")

# === ✅ Download images into master folder
print(f"\n⬇️  Downloading {len(subset_df)} images to: {MASTER_FOLDER}")

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Chrome')]
urllib.request.install_opener(opener)

for ix, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
    url = row['url']
    filename = row['md5hash']
    dest_path = os.path.join(MASTER_FOLDER, filename)

    if not os.path.exists(dest_path):
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as e:
            print(f"❌ Failed to download {url}: {e}")

print("\n✅ Done downloading all available images.")
