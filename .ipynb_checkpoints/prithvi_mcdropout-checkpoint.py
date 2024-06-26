"""
this python script applies montecarlo dropout to the prithvi foundation model's inference on the sen1floods11 flood segmentation dataset. 

it writes results to a json file called metrics.json.

ARGS: 
--gpu (include this flag for inference with gpu.)
--stop (int) (include the flag with an integer n, to stop inference after image n)
--mc (int) (specify number of montecarlo dropout trials for certainty estimation. default is 3.)

e.g.
python prithvi_mcdropout.py --gpu True --stop 2 --mc 2
"""
import argparse
import os
import socket
import torch
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import yaml
from mmcv import Config
from mmseg.apis import init_segmentor
from model_inference import inference_segmentor, process_test_pipeline
from huggingface_hub import hf_hub_download
import random
from scipy import stats
import matplotlib
import json
from PIL import Image
import sys

def load_raster(path, crop=None):
    with rasterio.open(path) as src:
        img = src.read()
        # load first 6 bands
        img = img[:6]
        img = np.where(img == NO_DATA, NO_DATA_FLOAT, img)
        if crop:
            img = img[:, -crop[0]:, -crop[1]:]
    return img

def enhance_raster_for_visualization(raster, ref_img=None):
    if ref_img is None:
        ref_img = raster
    channels = []
    for channel in range(raster.shape[0]):
        valid_mask = np.ones_like(ref_img[channel], dtype=bool)
        valid_mask[ref_img[channel] == NO_DATA_FLOAT] = False
        mins, maxs = np.percentile(ref_img[channel][valid_mask], PERCENTILES)
        normalized_raster = (raster[channel] - mins) / (maxs - mins)
        normalized_raster[~valid_mask] = 0
        clipped = np.clip(normalized_raster, 0, 1)
        channels.append(clipped)
    clipped = np.stack(channels)
    channels_last = np.moveaxis(clipped, 0, -1)[..., :3]
    rgb = channels_last[..., ::-1]
    return rgb

def reg_inference(model, image_path, gpu):
    """function to run inference on the prithvi model with any given images. make sure orignal_model/model.pth exists. this is an original copy of the finetuned model"""
    device="cpu"
    if gpu:
        device="cuda"


    copy_model = torch.load('original_model/model.pth', map_location=torch.device(device))
    copy_model.load_state_dict(model.state_dict())
    copy_model.eval()
    copy_test_pipeline = process_test_pipeline(copy_model.cfg.data.test.pipeline)
    result = inference_segmentor(copy_model, image_path, custom_test_pipeline=copy_test_pipeline)
    return result



def enable_dropout(model, drop):
    """ function to enable the dropout layers during test-time, along with adding changing the dropout rate based on a randomized parameter. """

    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            m.p = drop
            

def get_monte_carlo_predictions(model, forward_passes, image_path, gpu):
    """ function to perform inference with monte carlo dropout. calls enable_dropout function and returns monte carlo dropout predictions."""

    device = "cpu"
    if gpu:
        device="cuda"
    # load the model from disk and load state dict to avoid changing the original model
    copy_model = torch.load('original_model/model.pth', map_location=torch.device(device))
    copy_model.load_state_dict(model.state_dict())
    copy_model.train()  
    predictions = []

    # inference
    for _ in range(forward_passes):
        enable_dropout(copy_model, drop=random.uniform(0, 0.5))
        copy_test_pipeline = process_test_pipeline(copy_model.cfg.data.test.pipeline)
        result = inference_segmentor(copy_model, image_path, custom_test_pipeline=copy_test_pipeline)
        predictions.append(result)
        
    return predictions
  


def heatmap(preds):
    """given n-arrays, this function generates an in-line heat map of the total stacked variance of each position in the array. Will also return heatmap array."""

    stacked_arrays = np.stack([arr[0] for arr in preds], axis=0)
    variance_array = np.var(stacked_arrays, axis=0)
    
    mode_array = stats.mode(stacked_arrays, axis=0)[0]

    return variance_array, mode_array




def eval_certainty(mode_array, groundtruth, orig):
    """this function evaluates specified certainty pixels against ground truth labels. it will also compute IoU, mIoU, F1, """

    
    results = {"Original_IoU" : 0, "Original_mIoU" : 0, "Original_F1" : 0, "Original_mF1": 0, "MC_IoU": 0, "MC_mIoU" : 0, "MC_F1":0, "MC_mF1":0, "MC_recall": 0, "Original_recall":0}
    groundtruth = load_raster(groundtruth)[0]
    mode_array = np.where(mode_array == -1, 0, mode_array)
    groundtruth = np.where(groundtruth == -1, 0, groundtruth)


    correct_matches = np.sum(np.logical_and(orig[0] == 1, groundtruth == 1))
    total_elements = np.sum(groundtruth == 1)
    original_IoU = correct_matches / total_elements
    results["Original_IoU"] = original_IoU

    # MC IoU calculation 
    correct_matches = np.sum(np.logical_and(mode_array == 1, groundtruth == 1))
    certainty_IoU = correct_matches / total_elements
    results["MC_IoU"] = certainty_IoU

    # mIoU calculations (IoU for all classes)
    correct_matches = np.sum(orig[0] == groundtruth)
    total_elements = np.prod(groundtruth.shape)
    original_mIoU = correct_matches / total_elements
    results["Original_mIoU"] = original_mIoU

    correct_matches = np.sum(mode_array == groundtruth)
    certainty_mIoU = correct_matches / total_elements
    results["MC_mIoU"] = certainty_mIoU


    

    

    # # Original F1 calculation
    # original_tp = np.sum(np.logical_and(orig[0] == 1, orig[0] == 1))
    # original_fp = np.sum(np.logical_and(orig[0] != ground_truth, orig[0] == 1)
    # precision = original_tp/ original_tp + original+fp
    # recall = original_tp / np.sum(groundtruth == 1)
    
    # original_F1 = 2 * (precision * recall) / (precision + recall)
    # results["Original_F1"] = original_F1
    
    # # MC F1 calculation
    # certainty_F1 = 2 * (precision * recall) / (precision + recall)
    # results["MC_F1"] = certainty_F1
    
    # # mF1 calculations (F1 for all classes)
    # mc_tp = np.sum(np.logical_and(orig[0] == 1, orig[0] == 1))
    # mc_fp = np.sum(np.logical_and(orig[0] != ground_truth, orig[0] == 1)
    # precision = mc_tp / (mc_tp + mc_fp)
    # recall = mc_tp / np.sum(groundtruth == 1)
    
    # # Original mF1
    # original_mF1 = 2 * (precision * recall) / (precision + recall)
    # results["Original_mF1"] = original_mF1
    
    # # MC mF1
    # certainty_mF1 = 2 * (precision * recall) / (precision + recall)
    # results["MC_mF1"] = certainty_mF1
    
    
    


    
    return results 


if __name__ == "__main__":
    device="cpu"
    NO_DATA = -9999
    NO_DATA_FLOAT = 0.0001
    PERCENTILES = (0.1, 99.9)
    MULTIPLE_GPU = False
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='Use GPU or not')
    parser.add_argument('--stop', type=int, default=-1, help='Early stopping mechanism during inference')
    parser.add_argument('--mc', type=int, default=3, help="Certainty simulation count")

    args = parser.parse_args()


    # TODO: implement multiple gpu inference
    if args.gpu == 1:
        print("Using GPU")
        device="cuda"
    elif args.gpu>1:
        MULTIPLE_GPU = True
        print("Using multiple GPU")
        device="cuda"
    else:
        print("Not using GPU")

    if args.stop== -1:
        print("No early stopping.")
    elif args.stop>0:
        print(f"Stopping inference on image {args.stop}")
    else:
        print("Invalid stopping count. --stop must be >=1")
        sys.exit(1)

    if args.mc < 2:
        print("Invalid simulation count. --mc must be >=2")
        sys.exit(1)
    else:
        print(f"Simulation count == {args.mc}")



    directory = "test_labels"
    if os.path.exists(directory) == False:
        print(f"Error: The directory '{directory}' does not exist.", file=sys.stderr)
        print(f"Please create {directory} with labeled flood segmentation images inside (..._LabelHand.tif).", file=sys.stderr)
        sys.exit(1)


    # TO DO: add try and except for model download
    # download finetuned model weights and config from huggingface
    config_path=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename="sen1floods11_Prithvi_100M.py")
    ckpt=hf_hub_download(repo_id="ibm-nasa-geospatial/Prithvi-100M-sen1floods11", filename='sen1floods11_Prithvi_100M.pth')
    finetuned_model = init_segmentor(Config.fromfile(config_path), ckpt, device)
    

    if os.path.exists("original_model/model.pth") == False:
        os.mkdir("original_model")
        torch.save(finetuned_model, "original_model/model.pth")


    images={}
    metrics={}
    
    # early stopping for testing
    stop_count = 0


    # iterate through a directory and use its images as inference, specify ground truth path inside the loop
    for filename in os.listdir(directory):
        if args.stop > 0 and stop_count == args.stop: 
            print("Early stop.")
            break



        imgname = filename.split("_")
        imgname=f"{imgname[0]}_{imgname[1]}_S2Hand.tif"
        filepath = os.path.join(directory, filename)

        
        ground_truth_path = filepath
        # use the testing image directory
        image_path = f"test_images/{imgname}"
        # image_path = f"v1.1/data/flood_events/HandLabeled/S2Hand/{imgname}"

        if os.path.exists(image_path) == False:
            print(f"Error: The path '{image_path}' does not exist.", file=sys.stderr)
            print(f"Please ensure the Sen1Floods11 datasetis present in the workspace.", file=sys.stderr)
            print("It can be installed using: gsutil -m rsync -r gs://sen1floods11 .")
            sys.exit(1)

        
        print(f"Image: {image_path} \nLabel: {filepath}")

        # inference with Prithvi and certainty estimation + evaluation
        orig = reg_inference(finetuned_model, image_path, args.gpu)
        mc_preds = get_monte_carlo_predictions(finetuned_model, args.mc, image_path, args.gpu)
        arr, mode_arr=heatmap(mc_preds)
        results = eval_certainty(mode_arr, ground_truth_path, orig)

        metrics[image_path] = results
        images[image_path] = {"original_image": orig[0], "certainty_estimate":arr, "mode_arr": mode_arr}

        print(results)
        stop_count+=1

        # write data to file 
        json_data = json.dumps(metrics)
        with open(f"metrics_2{args.mc}.json", 'w') as json_file:
            json_file.write(json_data)




    # OPTIONAL - IMAGE SAVING

        base_dir = "inference_images2"

        os.makedirs(base_dir, exist_ok=True)

        image_folder = os.path.join(base_dir, os.path.splitext(os.path.basename(image_path))[0])
        os.makedirs(image_folder, exist_ok=True)

    
        mode_arr = mode_arr.astype(np.uint8)


        
        plt.imsave(os.path.join(image_folder, "original_image.jpg"), enhance_raster_for_visualization(load_raster(image_path)))
    
        plt.imsave(os.path.join(image_folder, "ground_truth.jpg"), load_raster(ground_truth_path)[0])

        
        plt.imsave(os.path.join(image_folder, "original_pred.jpg"), orig[0])
        
        # Save certainty_estimate image with viridis color map
        plt.imsave(os.path.join(image_folder, f"certainty_estimate_viridis{args.mc}.jpg"), arr, cmap='viridis')
        
        # Save mode_arr image with viridis color map
        plt.imsave(os.path.join(image_folder, f"mode_arr_viridis{args.mc}.jpg"), mode_arr, cmap='viridis')
