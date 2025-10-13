import argparse
import sys
from pathlib import Path
from shutil import copy

import librosa
import numpy as np
import pandas as pd
import random
from os import listdir
from os.path import join

import config

# Ignore excessive warnings
import logging
logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)

# Set random seeds and deterministic pytorch for reproducibility
random.seed(config.SEED)       # python random seed
np.random.seed(config.SEED)    # numpy random seedimport numpy as np


def remove_non_eligible_patients(clinical_data_df):
    """Remove patients with:
        - uninterpretable spirometry
        and
        - with pathology 'atteinte des petites voies respiratoires'
    """
    selected_clinical_data_df = clinical_data_df.copy()

    # Remove patients with 'spiro' equals to:
    # - NaN
    # - 3 (meaning 'non interpretable')
    selected_clinical_data_df = selected_clinical_data_df[selected_clinical_data_df['spiro'].isin([1, 2])]

    # Remove patient with "Atteinte des petites voies respiratoires"
    selected_clinical_data_df = selected_clinical_data_df[selected_clinical_data_df['pathology'] != 2]

    return selected_clinical_data_df

def add_asthma_target_variable(clinical_data_df):
    df = clinical_data_df.copy()

    # Define target variable 'asthma'
    # ('spiro' is encoded as {1: pathologic, 2: non-pathologic})
    df['asthma'] = df['spiro'].replace({2: 0})

    return df


def get_preprocessed_clinical_data(features_dirpath,
                                   clinical_data_and_data_dictionary_filepaths_dict={}):
    clinical_data_filepath = clinical_data_and_data_dictionary_filepaths_dict['Ap']['clinical_data']
    data_dictionary_filepath = clinical_data_and_data_dictionary_filepaths_dict['Ap']['data_dictionary']

    clinical_data_df = pd.read_csv(clinical_data_filepath)
    data_dictionary_df = pd.read_csv(data_dictionary_filepath)
    print(clinical_data_filepath)

    # # Add additional variables and description to categorical variables
    # clinical_data_df = preprocess_df(clinical_data_df, data_dictionary_df)

    # # Homogenize 'patient_id' variable (and its derived components, e.g. study)
    # clinical_data_df = homogenize_components_of_df(clinical_data_df)

    return clinical_data_df


def prepare_data(root_path, project_locations,
                 clinical_data_and_data_dictionary_filepaths_dict={},
                 recordings_dirpath="../data", n_splits=5,
                 verbose=False, copy_files=False):
    """Return patient_df containing included patients,
    and copy their recordings in data directory.

    Notes:
    - the folds attributed to each patient should not change if we simply add new
    patients at the end of the clinical data file.
    - if a patient in the middle of the clinical data file is removed, the folds of
    the patients following it and being in the same asthma group will be shifted by one
    """
    # Create directory that will contain the audio recordings
    Path(recordings_dirpath).mkdir(parents=True, exist_ok=True)

    patient_df = []

    for project, locations in project_locations.items():
        for location in locations:
            print(f"{project}: {location}")
            data_folder = join(root_path, f"{project}_{location}")

            # Get all available wav filenames
            wav_filenames = sorted([filename for filename in listdir(data_folder) if filename.endswith('.wav')])

            # Load and preprocess the clinical data file
            clinical_db = get_preprocessed_clinical_data(
                join(root_path, "features"),
                clinical_data_and_data_dictionary_filepaths_dict)

            # Load and preprocess the clinical data file
            # clinical_db = pd.read_csv(clinical_path)
            clinical_db = remove_non_eligible_patients(clinical_db)
            clinical_db = add_asthma_target_variable(clinical_db)

            # Select and copy recordings of each patient, and add patient to patient_df
            for _, r in clinical_db.iterrows():
                r['patient'] = r['patient_id_wo_stetho']
                r['location'] = location

                # Get patient's recording files
                patient_asthma = r['asthma']
                patient_wav_files = [filename for filename in wav_filenames if filename.startswith(r['patient'])]

                # If patient has asthma, do _not_ include the recordings taken after (post) ventolin
                if patient_asthma == 1:
                    patient_wav_files = [filename for filename in patient_wav_files if filename.split('.')[0].endswith('a')]

                # Copy all selected patient files into the 'data' directory
                for patient_wav_filename in patient_wav_files:
                    audio_file = join(data_folder, patient_wav_filename)

                    if copy_files:
                        copy(audio_file, recordings_dirpath)
                        if verbose:
                            print(f"{audio_file}\ncopied to\n{recordings_dirpath}")

                patient_df.append(r)

    # Add fold to each patient
    patient_df = pd.DataFrame(patient_df)
    grouped_patients = patient_df.groupby(["location", "asthma"])["patient"].apply(list).reset_index()

    df_with_folds = []
    for _, r in grouped_patients.iterrows():
        base_fold = np.random.randint(n_splits)
        for i, patient in enumerate(r["patient"]):
            patient_fold = (base_fold + i) % n_splits
            row = {
                "patient": patient,
                "location": r["location"],
                "fold": patient_fold,
            }
            df_with_folds.append(row)

    patient_df = pd.DataFrame(df_with_folds).merge(patient_df, on=["patient", "location"])

    return patient_df

def get_samples(patient_df, out_path, audio_array_path, audio_meta_path,
                selected_stethos=["L", "E"], asthmoscope=True):
    sample_length = config.MAX_DURATION * config.RATE

    audio_dataset = []
    samples_df = []

    # Get all recording filenames
    wav_filenames = sorted({filename for filename in listdir(out_path) if filename.endswith('.wav')})

    for _, r in patient_df.iterrows():
        audio_files = sorted([filename for filename in wav_filenames if filename.startswith(r['patient'])])

        for f in audio_files:
            filepath = join(out_path, f)
            samples, sr = librosa.load(filepath, sr=None, duration=config.MAX_DURATION)
            n_samples = samples.size
            assert sr == config.RATE

            # Get relevant info from filename
            code = f.split('.')[0]
            new_sample = dict(r)
            new_sample["file"] = filepath
            new_sample["position"] = code.split('_')[5]
            new_sample["stethoscope"] = code.split('_')[4]
            new_sample["ventolin_ap"] = code.split('_')[7]
            new_sample["site"] = code.split('_')[6][0]
            new_sample["session_number"] = code.split('_')[6][1]
            new_sample["ventolin_ap"] = code.split('_')[7]
            new_sample["patient_session_id"] = "_".join([
                new_sample["patient"],
                new_sample["stethoscope"],
                new_sample["site"] + new_sample["session_number"],
                new_sample["ventolin_ap"],
            ])
            new_sample["end"] = n_samples
            new_sample["duration"] = n_samples / config.RATE

            # Only include sample if it is one of the selected stethos
            if new_sample["stethoscope"] in selected_stethos:
                samples = librosa.util.fix_length(samples, size=sample_length)
                audio_dataset.append(samples)
                samples_df.append(new_sample)

    audio_dataset = np.array(audio_dataset)
    np.save(audio_array_path, audio_dataset)

    samples_df = pd.DataFrame(samples_df)
    samples_df.to_csv(audio_meta_path, index=False)
