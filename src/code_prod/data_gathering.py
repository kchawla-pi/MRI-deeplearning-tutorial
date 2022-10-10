from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from numpy import ndarray
from pandas import DataFrame, Series
from skimage.io import imsave
from tqdm import tqdm


def save_excel_as_csv(excel_filepath: Path, csv_filepath: Path, overwrite: bool = False) -> DataFrame:
    if overwrite or not csv_filepath.exists():
        excel_as_csv = pd.read_excel(
            io=excel_filepath,
            engine="openpyxl",
            usecols=list(range(4)),
            )
        excel_as_csv.to_csv(csv_filepath)
        return excel_as_csv


@lru_cache
def load_image_uid_paths_mappings(excel_filepath) -> DataFrame:
    csv_filepath = excel_filepath.with_suffix(".csv")
    csv_data = save_excel_as_csv(excel_filepath=excel_filepath, csv_filepath=csv_filepath)
    if csv_data is None:
        csv_data = pd.read_csv(csv_filepath)
    return csv_data


def filter_map_fat_saturated_scans_first_100_patients(image_uid_to_paths_map) -> DataFrame:
    fat_saturated_scans_map = image_uid_to_paths_map[image_uid_to_paths_map["original_path_and_filename"].str.contains("pre")]
    first_100_patients_filtering_pattern = "|".join([f"DICOM_Images/Breast_MRI_{patient_num:03d}" for patient_num in range(101)])
    first_100_patients_map = fat_saturated_scans_map[fat_saturated_scans_map["original_path_and_filename"].str.contains(first_100_patients_filtering_pattern)]
    return first_100_patients_map


def convert_patient_dicom_to_png(
    patient_record: Series,
    bounding_boxes: DataFrame,
    images_dirpath: Path,
    destination_image_dirpath: Path,
    n_cancer_positive_extracted_per_class: int,
    n_cancer_negative_extracted_per_class: int,
    n_samples_per_class: int,
    ):
    patient_id = extract_patient_id(patient_record['original_path_and_filename'])
    tumor_bounding_box = bounding_boxes[bounding_boxes["Patient ID"] == patient_id]
    slice_within_tumor_boundary = is_slice_within_tumor_boundary(
        tumor_bounding_box=tumor_bounding_box,
        slice_idx=extract_slice_idx(patient_record['original_path_and_filename']),
        allowance=5,
        )
    dicom_filepath = images_dirpath / patient_record['classic_path']
    if slice_within_tumor_boundary and n_cancer_positive_extracted_per_class <= n_samples_per_class:
        n_cancer_positive_extracted_per_class += 1
        save_dicom_as_png(
            dicom_filepath=dicom_filepath,
            save_dirpath=destination_image_dirpath,
            images_dirpath=images_dirpath,
            tumor_positive=slice_within_tumor_boundary,
            )

    elif not slice_within_tumor_boundary and n_cancer_negative_extracted_per_class <= n_samples_per_class:
        n_cancer_negative_extracted_per_class += 1
        save_dicom_as_png(
            dicom_filepath=dicom_filepath,
            save_dirpath=destination_image_dirpath,
            images_dirpath=images_dirpath,
            tumor_positive=slice_within_tumor_boundary,
            )


def extract_patient_id(original_image_path: str) -> str:
    return "_".join(Path(original_image_path).stem.split("_")[:3])


def extract_slice_idx(original_image_path: str) -> int:
    return int(Path(original_image_path).stem.split("_")[-1])


def is_slice_within_tumor_boundary(tumor_bounding_box, slice_idx, allowance) -> bool:
    return (
        (slice_idx > int(tumor_bounding_box["Start Slice"]) - allowance)
        and
        (slice_idx < int(tumor_bounding_box["End Slice"]) + allowance)
    )


def save_dicom_as_png(
    dicom_filepath: Path,
    save_dirpath: Path,
    images_dirpath: Path,
    tumor_positive:bool,
    ):
    destination_image_filepath = make_save_paths(
        images_dirpath=images_dirpath,
        save_dirpath=save_dirpath,
        dicom_filepath=dicom_filepath,
        tumor_positive=tumor_positive,
        )
    if not destination_image_filepath.exists():
        dicom_to_png(
            dicom_filepath=dicom_filepath,
            png_filepath=destination_image_filepath,
            )


def dicom_to_png(dicom_filepath, png_filepath):
    dicom_image = pydicom.dcmread(dicom_filepath)
    dicom_pixels = normalize_to_uint8(data=dicom_image.pixel_array)
    if dicom_image.PhotometricInterpretation == "MONOCHROME1":
        dicom_pixels = np.invert(dicom_pixels)
    imsave(png_filepath.resolve(), dicom_pixels)


def normalize_to_uint8(data: ndarray):
    return (data * 255 / data.max()).astype("uint8")


def make_save_paths(
    save_dirpath: Path,
    images_dirpath: Path,
    dicom_filepath: Path,
    tumor_positive: bool,
    ):
    save_path = (
        save_dirpath
        / f"tumor_{'positive' if tumor_positive else 'negative'}"
        / dicom_filepath.relative_to(images_dirpath)
    ).with_suffix(".png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def main():
    data_dirpath = Path("data")

    bounding_boxes_filepath = data_dirpath / "Annotation_Boxes.xlsx"
    image_uid_to_paths_map_filepath_xls = data_dirpath / "Breast-Cancer-MRI-filepath_filename-mapping.xlsx"

    bounding_boxes = pd.read_excel(bounding_boxes_filepath)

    image_uid_to_paths_map = load_image_uid_paths_mappings(image_uid_to_paths_map_filepath_xls)
    first_100_patients_map = filter_map_fat_saturated_scans_first_100_patients(image_uid_to_paths_map)

    n_samples_per_class = 2600
    n_cancer_negative_extracted_per_class = 0
    n_cancer_positive_extracted_per_class = 0

    images_dirpath = data_dirpath / "manifest-1664908432813"
    destination_image_dirpath = data_dirpath / "png_out"

    for _, patient_record in tqdm(first_100_patients_map.iterrows(), total=n_samples_per_class*2):
        convert_patient_dicom_to_png(
            patient_record,
            bounding_boxes,
            images_dirpath,
            destination_image_dirpath,
            n_cancer_positive_extracted_per_class,
            n_cancer_negative_extracted_per_class,
            n_samples_per_class,
            )

        ...


if __name__ == "__main__":
    main()
