import os

import numpy as np
import torch
from tqdm import tqdm

from data.util import custom_collate_batch


def flownet_batch(batch, model):
    previous = []
    current = []
    previous.append(batch[0][0][0][:, :, :3].to(model.device))  # Lidar properties are not needed
    previous.append(batch[0][0][1].to(model.device))
    previous.append(batch[0][0][2].to(model.device))
    current.append(batch[0][1][0][:, :, :3].to(model.device))  # Lidar properties are not needed
    current.append(batch[0][1][1].to(model.device))
    current.append(batch[0][1][2].to(model.device))
    flows = batch[1].to(model.device)
    new_batch = ((previous, current), flows)
    return new_batch

def predict_flows(model, dataset, offset, architecture):
    if architecture == "FlowNet":
        dataset.pillarize(False)
        (previous_frame, current_frame), flows = dataset[offset]
        # We set batchsize of 1 for predictions
        batch = custom_collate_batch([((previous_frame, current_frame), flows)])
        batch = flownet_batch(batch, model)
        with torch.no_grad():
            output = model(batch[0])
        predicted_flows = output[0].data.cpu().numpy()
        return predicted_flows
    elif architecture == "FastFlowNet":  # This model always uses GPU
        dataset.pillarize(True)
        (previous_frame, current_frame), flows = dataset[offset]
        batch = custom_collate_batch([((previous_frame, current_frame), flows)])
        with torch.no_grad():
            output = model(batch[0])
        predicted_flows = output[0].data.cpu().numpy()
        return predicted_flows
    else:
        print(f"Architecture {architecture} not implemented")
        exit(1)


def get_flows(dataset, index):
    flows_folder = os.path.join(dataset.data_path, "flows")
    flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(index))
    predicted_flows = np.load(flows_name)['flows']
    return predicted_flows


def flows_exist(dataset):
    flows_folder = os.path.join(dataset.data_path, "flows")
    check_existing_folder = os.path.isdir(flows_folder)

    # If folder doesn't exist, then create it.
    if not check_existing_folder:
        return False

    else:
        print(f"Flows already exist, please remove {flows_folder} to process again the flows")
        print("Using already predicted flows...")
        return True

def predict_and_store_flows(model, dataset, architecture):
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
        predicted_flows = predict_flows(model, dataset, i, architecture)
        flows_name = os.path.join(flows_folder, "flows_" + dataset.get_name_current_frame(i))
        np.savez_compressed(flows_name, flows=predicted_flows)