import os
import argparse
from tqdm import tqdm
import numpy as np
from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from os.path import join as opj
from dataset import VOC2012Dataset, VOC_CLASSNAMES
from metrics import IoUMetrics
from prompter import CLIPPrompter, load_clip_to_cpu

from logger import get_logger


def train_one_epoch(train_dl, opt, models):
    sam: Sam = models["sam"]
    prompter: CLIPPrompter = models["prompter"]

    prompter.train()
    tqdm_loader = tqdm(train_dl)
    seg_loss_fn = nn.CrossEntropyLoss(ignore_index=255)

    sparse_emb, dense_emb = sam.prompt_encoder(None, None, None)
    dense_posemb = sam.prompt_encoder.get_dense_pe()

    iou_metrics = IoUMetrics(class_names=["background", "foreground"])
    per_class_iou_metrics = IoUMetrics(num_classes=21, class_names=["background"] + VOC_CLASSNAMES)
    for emb, class_indices, semseg, image in tqdm_loader:
        emb = emb.cuda()
        class_indices = class_indices.cuda()
        semseg = semseg.cuda()

        if prompter.trainer == "coop":
            text_prompts = prompter(class_indices)
        else:
            image = image.cuda()
            text_prompts = prompter(class_indices, image)

        outputs = []
        output_ious = []
        gt_ious = []
        for bi in range(emb.size(0)):
            sparse_embeddings = torch.cat([
                sparse_emb,
                text_prompts[bi].unsqueeze(0)
            ], dim=1)

            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=emb[bi],
                image_pe=dense_posemb,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            low_res_masks = low_res_masks[0]
            iou_predictions = iou_predictions[0]

            match_index = torch.argmax(iou_predictions)

            match_mask = low_res_masks[match_index]
            match_iou = iou_predictions[match_index]
            outputs.append(torch.stack(
                [torch.zeros_like(match_mask), match_mask], dim=0))
            output_ious.append(match_iou)

            non_ignore = semseg[bi] != 255
            mask = match_mask[non_ignore] > 0
            gt = semseg[bi][non_ignore] == 1

            gt_ious.append((mask & gt).sum() / ((mask | gt).sum() + 1e-6))

        pred_logits = torch.stack(outputs, dim=0)
        output_ious = torch.stack(output_ious, dim=0)
        gt_ious = torch.stack(gt_ious, dim=0)

        iou_loss = F.mse_loss(output_ious, gt_ious)
        seg_loss = seg_loss_fn(pred_logits, semseg)
        total_loss = seg_loss + iou_loss

        opt.zero_grad()
        total_loss.backward()
        opt.step()

        # Update IoU
        pred_label = torch.argmax(pred_logits, dim=1)
        iou_metrics.update(pred_label, semseg)

        ci = class_indices[:, None, None] + 1
        semseg = torch.where(semseg==1, ci, semseg)
        per_class_iou_metrics.update(pred_label*ci, semseg)

        tqdm_loader.set_description(
            f"{seg_loss:.6f} {iou_loss:.6f} {iou_metrics} {per_class_iou_metrics.miou}")

    logger.info(iou_metrics)
    logger.info(per_class_iou_metrics)
    return iou_metrics.ious


@torch.no_grad()
def validate(val_dl, models):
    sam: Sam = models["sam"]
    prompter: CLIPPrompter = models["prompter"]

    prompter.eval()
    tqdm_loader = tqdm(val_dl)

    sparse_emb, dense_emb = sam.prompt_encoder(None, None, None)
    dense_posemb = sam.prompt_encoder.get_dense_pe()

    iou_metrics = IoUMetrics(class_names=["background", "foreground"])
    per_class_iou_metrics = IoUMetrics(num_classes=21, class_names=["background"] + VOC_CLASSNAMES)
    for emb, class_indices, semseg, image in tqdm_loader:
        emb = emb.cuda()
        class_indices = class_indices.cuda()
        semseg = semseg.cuda()

        if prompter.trainer == "coop":
            text_prompts = prompter(class_indices)
        else:
            image = image.cuda()
            text_prompts = prompter(class_indices, image)

        outputs = []
        for bi in range(emb.size(0)):
            sparse_embeddings = torch.cat([
                sparse_emb,
                text_prompts[bi].unsqueeze(0)
            ], dim=1)

            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=emb[bi],
                image_pe=dense_posemb,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )

            low_res_masks = low_res_masks[0]
            iou_predictions = iou_predictions[0]

            index = torch.argmax(iou_predictions)

            match_mask = low_res_masks[index]
            outputs.append(torch.stack(
                [torch.zeros_like(match_mask), match_mask], dim=0))

        pred_logits = torch.stack(outputs, dim=0)
        pred_label = torch.argmax(pred_logits, dim=1)

        iou_metrics.update(pred_label, semseg)

        ci = class_indices[:, None, None] + 1
        semseg = torch.where(semseg==1, ci, semseg)
        per_class_iou_metrics.update(pred_label*ci, semseg)
        tqdm_loader.set_description(f"{iou_metrics} {per_class_iou_metrics.miou}")

    logger.info(iou_metrics)
    logger.info(per_class_iou_metrics)
    logger.info("\n")
    return iou_metrics.ious


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str,
                        default="./pretrain-weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voc2012-path", type=str, default="./VOC2012")
    parser.add_argument("--n-emb", type=int, required=True)
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--trainer", type=str, default="coop")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--epoch", type=float, default=100)
    parser.add_argument("--unseen-classes", nargs='+',
                        type=int, default=[17, 18, 19, 20])
    
    parser.add_argument("--save_dir", type=str, default='./ckpt')

    args = parser.parse_args()
    return args


def get_dataloader(args):
    embeddings_folder = opj(args.voc2012_path, f"Embeddings_{args.model_type}")
    split_folder = opj(args.voc2012_path, "ImageSets", "Segmentation")

    batch_size = args.batch_size
    num_workers = args.num_workers

    unseen_classes = args.unseen_classes
    seen_classes = [
        i
        for i in range(1, 21)
        if i not in unseen_classes
    ]

    train_ds = VOC2012Dataset(args.voc2012_path,
                              opj(split_folder, "train.txt"),
                              embeddings_folder,
                              ignore_classes=unseen_classes)
    val_ds = VOC2012Dataset(args.voc2012_path,
                            opj(split_folder, "val.txt"),
                            embeddings_folder)

    train_dl = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, num_workers=num_workers)
    val_dl = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=num_workers)

    return train_dl, val_dl


def create_models(device, args):
    checkpoint_path = args.checkpoint_path
    model_type = args.model_type
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.prompt_encoder.eval().to(device)
    sam.mask_decoder.eval().to(device)

    clip_rn50 = load_clip_to_cpu("RN50")
    prompter = CLIPPrompter(args.n_emb, VOC_CLASSNAMES, clip_rn50, trainer=args.trainer).to(device)

    return {
        "sam": sam,
        "prompter": prompter,
    }


def save_checkpoint(models, optimizer, epoch, best_miou, ckpt_path):
    state = {
        "sam":  models['sam'].state_dict(),
        "prompter":  models['prompter'].state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_miou": best_miou,
    }
    logger.info(f'Saving model: {ckpt_path}')
    torch.save(state, ckpt_path)


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.save_dir, "log.txt"))

    logger.info(args)

    device = args.device

    train_dl, val_dl = get_dataloader(args)

    models = create_models("cuda", args)
    sam = models["sam"]
    prompter = models["prompter"]

    opt = torch.optim.Adam(prompter.parameters(), args.lr)

    best_epoch = 0
    best_iou = 0
    for epoch in range(args.epoch):
        logger.info(f"epoch [{epoch}]")
        train_ious = train_one_epoch(train_dl, opt, models)
        val_ious = validate(val_dl, models)

        if val_ious[1] > best_iou:
            best_iou = val_ious[1]
            best_epoch = epoch
            ckpt_path = os.path.join(args.save_dir, 'best.pth')
            save_checkpoint(models, opt, epoch, best_iou, ckpt_path)
            logger.info(f"epoch [{best_epoch}] has the best iou: {best_iou:.6f}")
    
    ckpt_path = os.path.join(args.save_dir, 'last.pth')
    save_checkpoint(models, opt, args.epoch - 1, val_ious[1], ckpt_path)
