import numpy as np
import os
from tqdm import tqdm

# Paths to input and output directories
rmf_input_dir = "/mnt/g/rmf/rmf_superpixel_dataset/tile_128"
wrf_input_dir = "/mnt/g/wrf/wrf_superpixel_dataset/tile_128"
# ovf_input_dir = "/mnt/g/ovf/ovf_superpixel_dataset/tile_128"

# Overlapping species between two datasets
overlap_species = ["BF", "BW", "LA", "PT", "PJ", "SB", "SW"]

# Original species lists for reference
rmf_species = ["BF", "BW", "CE", "LA", "PT", "PJ", "PO", "SB", "SW"]
wrf_species = ["SB", "LA", "PJ", "BW", "PT", "BF", "CW", "SW"]
# ovf_species = ['AB', 'PO', 'MR', 'BF', 'CE', 'PW', 'MH', 'BW', 'SW', 'OR', 'PR']

# Index mapping to reduce labels to 4-dimensions
rmf_indices = [rmf_species.index(sp) for sp in overlap_species]
wrf_indices = [wrf_species.index(sp) for sp in overlap_species]
# ovf_indices = [ovf_species.index(sp) for sp in overlap_species]

def filter_and_update(data, indices):
    label = data['label']

    other_species_proportion = np.sum(np.delete(label, indices))
    overlap_species_proportion = np.sum(label[indices])

    if other_species_proportion == 0 and np.isclose(overlap_species_proportion, 1.0, atol=1e-6):
        updated_label = label[indices]

        updated_data = {
            "superpixel_images": data["superpixel_images"],
            "point_cloud": data["point_cloud"],
            "label": updated_label,
            "per_pixel_labels": data["per_pixel_labels"][indices, :, :],
            "nodata_mask": data["nodata_mask"]
        }

        return updated_data
    else:
        return None

# Process and save data
def process_folder(input_dir, indices, dataset_name):
    splits = ["train", "test", "val"]
    for split in splits:
        split_input_dir = os.path.join(input_dir, split)
        output_dir = os.path.join(split_input_dir, "common")
        os.makedirs(output_dir, exist_ok=True)
        sp_input_dir = os.path.join(split_input_dir, f"{dataset_name}_sp")
        for filename in tqdm(os.listdir(sp_input_dir)):
            if filename.endswith('.npz'):
                data = np.load(os.path.join(sp_input_dir, filename), allow_pickle=True)
                data_dict = {key: data[key] for key in data.files}
                updated_data = filter_and_update(data_dict, indices)
                if updated_data:
                    np.savez(os.path.join(output_dir, filename), **updated_data)

# Execute processing
process_folder(rmf_input_dir, rmf_indices, "rmf")
process_folder(wrf_input_dir, wrf_indices, "wrf")
# process_folder(ovf_input_dir, ovf_indices)
