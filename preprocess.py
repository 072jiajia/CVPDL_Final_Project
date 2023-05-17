import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from os.path import join as opj


def preprocess(predictor: SamPredictor, images_folder, embeddings_folder, preprocess_files):
    for file in tqdm(preprocess_files):
        image_path = os.path.join(images_folder, f"{file}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out_path = os.path.join(embeddings_folder, file + ".npy")

        if not os.path.exists(out_path):
            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding().cpu().numpy()
            np.save(out_path, image_embedding)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str,
                        default="./sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", type=str, default="default")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voc2012-path", type=str, default="./VOC2012")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()

    images_folder = opj(args.voc2012_path, "JPEGImages")
    embeddings_folder = opj(args.voc2012_path, f"Embeddings_{args.model_type}")

    trainval_txt = opj(args.voc2012_path,
                       "ImageSets/Segmentation/trainval.txt")
    preprocess_files = [f.rstrip()
                        for f in open(trainval_txt, mode="r").readlines()]

    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
    sam.to(device=args.device)
    predictor = SamPredictor(sam)

    if not os.path.exists(embeddings_folder):
        os.makedirs(embeddings_folder)

    preprocess(predictor, images_folder, embeddings_folder, preprocess_files)
