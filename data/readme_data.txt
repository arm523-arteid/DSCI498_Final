This project uses healthcare data based on Synthea.  The data represents patient-level health records from
five U.S. states: California, Texas, Florida, New York, and Pennsylvania.
The focus is on detecting Myocardial Infarction (MI) events based on patient demographic and clinical features.

Two main datasets are provided:
train.csv
test.csv

These datasets were split using an 80/20 random split strategy.
Each patient has only one record summarizing their characteristics at a point in time.

Myocardial Infarction events are rare (~2.4% of patients flagged in training data).
This is important when interpreting model results and motivates the generative modeling work.

Full raw Synthea data is too large to store. The process_data.py script can re-generate processed files
if needed by downloading from public Google Drive links.
For basic model training and evaluation, use the preprocessed train.csv and test.csv provided under data/processed/.
Generated data is under the generated directory.

Below is the full structure of the data directory:

├── data/
│   ├── raw/:          This will remain empty. Given size, they reside in GoogleDrive.
│   ├── processed/:    Includes cleaned data ready for training
│   ├── generated/:    Data generated from VAE
│   └── readme_data.txt