import os
import argparse
import cv2
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from os.path import join as opj
from pathlib import Path
from pycocotools.coco import COCO

def preprocess_voc(predictor: SamPredictor, images_folder, embeddings_folder, preprocess_files):
    for file in tqdm(preprocess_files):
        image_path = os.path.join(images_folder, f"{file}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        out_path = os.path.join(embeddings_folder, file + ".npy")

        if not os.path.exists(out_path):
            predictor.set_image(image)
            image_embedding = predictor.get_image_embedding().cpu().numpy()
            np.save(out_path, image_embedding)

def preprocess_coco(predictor: SamPredictor, images_folder, embeddings_folder):
    img_dir = Path(images_folder)
    for img in tqdm(img_dir.glob("*.jpg")):
        file = str(img).split('/')[-1][:-4]

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
    parser.add_argument("--data_dir", type=str, default="./VOC2012")
    parser.add_argument("--dataset", type=str, default="VOC")
    args = parser.parse_args()
    return args
    

if __name__ == "__main__":
    args = parse_args()

    if args.dataset == 'VOC':
        images_folder = opj(args.data_dir, "JPEGImages")
        embeddings_folder = opj(args.data_dir, f"Embeddings_{args.model_type}")

        trainval_txt = opj(args.data_dir,
                        "ImageSets/Segmentation/trainval.txt")
        preprocess_files = [f.rstrip()
                            for f in open(trainval_txt, mode="r").readlines()]

        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
        sam.to(device=args.device)
        predictor = SamPredictor(sam)

        if not os.path.exists(embeddings_folder):
            os.makedirs(embeddings_folder)

        preprocess_voc(predictor, images_folder, embeddings_folder, preprocess_files)
    elif args.dataset == 'COCO':
        # path
        train_images_folder = opj(args.data_dir, "images/train2017")
        train_embed_folder = opj(args.data_dir, f"Embeddings_{args.model_type}/train")

        val_images_folder = opj(args.data_dir, "images/val2017")
        val_embed_folder = opj(args.data_dir, f"Embeddings_{args.model_type}/val")

        # SAM
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint_path)
        sam.to(device=args.device)
        predictor = SamPredictor(sam)

        # mkdir
        if not os.path.exists(train_embed_folder):
            os.makedirs(train_embed_folder)
        if not os.path.exists(val_embed_folder):
            os.makedirs(val_embed_folder)

        preprocess_coco(predictor, train_images_folder, train_embed_folder)
        preprocess_coco(predictor, val_images_folder, val_embed_folder)
    else:
        raise NotImplementedError
