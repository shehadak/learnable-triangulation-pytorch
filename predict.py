import argparse
import os
from glob import glob

import torch
from torch.utils.data import DataLoader

from mvn.models.triangulation import AlgebraicTriangulationNet
from mvn.utils import cfg
from mvn.utils import predict_utils as utils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, required=True, help="Path, where config file is stored"
    )
    parser.add_argument(
        "--videos_folder",
        type=str,
        default="val",
        help="Dataset split on which evaluate. Can be 'train' and 'val'",
    )

    args = parser.parse_args()
    return args


def setup_dataloader(videos_folder, intrinsic_params, extrinsic_params):
    video_paths = glob(os.path.join(videos_folder, "*.mp4"))
    dataset = utils.VideoDataset(video_paths, intrinsic_params, extrinsic_params)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )
    return dataloader


def predict(model, dataloader):
    """
    Perform inference on a given video using a pretrained model.

    Args:
    - model: Pretrained pose estimation model.
    - config: Configuration object.
    - dataloader: DataLoader supplying batches from the video.

    Returns:
    - keypoints_3d_all: Predicted 3D keypoints for all batches.
    - keypoints_2d_all: Predicted 2D keypoints for all batches.
    - heatmaps_all: Predicted heatmaps for all batches.
    - confidences_all: Predicted confidences for all batches.
    """

    model.eval()
    keypoints_3d_all, keypoints_2d_all, heatmaps_all, confidences_all = [], [], [], []

    # Loop through batches in the dataloader
    with torch.no_grad():
        for images_batch, proj_matricies_batch in dataloader:
            # Model inference
            (
                keypoints_3d_pred,
                keypoints_2d_pred,
                heatmaps_pred,
                confidences_pred,
            ) = model(images_batch, proj_matricies_batch, None)

            # Append predictions to lists
            keypoints_3d_all.append(keypoints_3d_pred.cpu())
            keypoints_2d_all.append(keypoints_2d_pred.cpu())
            heatmaps_all.append(heatmaps_pred.cpu())
            confidences_all.append(confidences_pred.cpu())

    # Convert lists to tensors
    keypoints_3d_all = torch.cat(keypoints_3d_all, dim=0)
    keypoints_2d_all = torch.cat(keypoints_2d_all, dim=0)
    heatmaps_all = torch.cat(heatmaps_all, dim=0)
    confidences_all = torch.cat(confidences_all, dim=0)

    return keypoints_3d_all, keypoints_2d_all, heatmaps_all, confidences_all


def main(args):
    device = torch.device(0)

    # config
    config = cfg.load_config(args.config)
    config.opt.n_iters_per_epoch = (
        config.opt.n_objects_per_epoch // config.opt.batch_size
    )

    model = AlgebraicTriangulationNet(config, device=device).to(device)

    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint)
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "")
            state_dict[new_key] = state_dict.pop(key)

        model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pre-trained weights for whole model")

    # datasets
    print("Loading data...")
    intrinsic_params, extrinsic_params = utils.params_from_images(config["calibration_video_path"], )
    loader = setup_dataloader(args.videos_folder, intrinsic_params, extrinsic_params)
    
    # experiment
    keypoints_3d_all, keypoints_2d_all, heatmaps_all, confidences_all = predict(
        model, loader
    )

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    print("args: {}".format(args))
    main(args)
