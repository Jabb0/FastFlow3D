import os

import numpy as np
import torch
from tqdm import tqdm

from data.util import custom_collate_batch


def predict_flows(model, dataset, offset):
    dataset.pillarize(True)
    (previous_frame, current_frame), flows = dataset[offset]
    # We set batchsize of 1 for predictions
    batch = custom_collate_batch([((previous_frame, current_frame), flows)])
    with torch.no_grad():
        output = model(batch[0])
    predicted_flows = output[0].data.cpu().numpy()
    return predicted_flows


def get_flows(dataset, index):
    flows_folder = os.path.join(dataset.data_path, "flows")
    flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(index))
    predicted_flows = np.load(flows_name)
    return predicted_flows


def predict_and_store_flows(model, dataset):
    flows_folder = os.path.join(dataset.data_path, "flows")
    check_existing_folder = os.path.isdir(flows_folder)

    # If folder doesn't exist, then create it.
    if not check_existing_folder:
        os.makedirs(flows_folder)
        print("created folder : ", flows_folder)

    else:
        print(f"Flows already exist, please remove {flows_folder} to process again the flows")
        print("Using already predicted flows...")
        return

    for i in tqdm(range(0, len(dataset)), desc="Predicting flows..."):
        predicted_flows = predict_flows(model, dataset, i)
        flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(i))
        np.save(flows_name, predicted_flows)