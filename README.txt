# DSCI498
This repo contains code and materials for a project that investigates the use of generative deep learning methods, 
specifically Variational Autoencoders (VAEs), to address limitations in structured healthcare data for heart 
failure (Myocardial Infrations) prediction.

The ultimate goal of this project is to enable the development of a classification model that accurately predicts
heart failure (MI) based on patient features. Generating higher-quality, balanced training data is a necessary step
toward building more reliable predictive models.

Heart failure is a complex and costly condition that often goes undetected until late stages. Predictive models can
offer early warning signals, but developing such models is challenging due to significant limitations in real-world
healthcare datasets, including strict privacy regulations, high acquisition costs, incomplete records, and a 
particularly low frequency of heart failure events. These challenges restrict the availability of sufficient data
needed for effective model training and evaluation.

To overcome these issues, this project begins with patient data generated using Synthea, an open-source
health record generator. Data from five states: California, Texas, Florida, New York, and Pennsylvania were
generated. These states were selected to provide a broad cross-section of urban, suburban, and rural populations and
to capture a diverse range of disease profiles. Nonetheless, heart failure events remain relatively rare, which
motivates the use of generative models to augment the training data. 

The primary aim of this project is to evaluate the effectiveness of VAEs models in generating
additional heart failure cases. These generative models allow for the augmentation of imbalanced datasets, which in
turn may improve the development of predictive models. A secondary aim is to explore the downstream performance of
simple classification models trained on datasets that include generated MI cases.

The project is structured as below:

MemajArteid
├── README.txt
├── data/
│   ├── raw/:            This will remain empty. Given size, they reside in GoogleDrive.
│   ├── processed/:      Includes cleaned data ready for training
│   ├── generated/:      Data generated from VAE
│   └── readme_data.txt
├── process_data.py      Reads data from GoogleDrive, cleans, creates cvd_flag and splits to train/test
├── ExploreData.ipynb    Provides exploratory visualizations and summaries of the data. (independent)
├── nn.py                Runs classification on original data
├── vae_train.py         Initial VAE
├── vae_train_augment.py Rerun of beta-VAE
├── nn_combined.py       Reruns classification on original plus generated data
├── main.py              Main - runs all programs

This project requires the following libraries:
-pandas
-numpy
-matplotlib
-seaborn
-torch (PyTorch)
-scikit-learn
-requests
-io
