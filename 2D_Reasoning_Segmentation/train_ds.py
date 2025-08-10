import os
import argparse
import shutil
import sys
import time
from functools import partial

import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)

def parse_args(args):
    parser = argparse.ArgumentParser(description="LISA Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    parser.add_argument(
        "--version", default="checkpoint/LISA-7B-v1-explanatory_base_new_hard_composite"
    )

    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)  
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument(
        "--dataset", default="reason_seg", type=str
    )
    parser.add_argument("--sample_rates", default="1", type=str)
    parser.add_argument(
        "--sem_seg_data",
        default="ade20k||cocostuff||pascal_part||paco_lvis||mapillary",
        type=str,
    )
    parser.add_argument(
        "--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog", type=str
    )
    parser.add_argument("--vqa_data", default="llava_instruct_150k", type=str)
    parser.add_argument("--reason_seg_data", default="Scannet_2D_Seg_base_new|train", type=str)  # "ReasonSeg|train", "Scannet_2D_Seg|train", "Scannet_2D_Seg_full|train"
    parser.add_argument("--val_dataset", default="Scannet_2D_Seg_base_new|val", type=str)  # "ReasonSeg|val", "Scannet_2D_Seg|val", "Scannet_2D_Seg_full|val"

    # parser.add_argument("--dataset_dir", default="/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/scannet_grounding", type=str)  # "./dataset"
    parser.add_argument("--dataset_dir", default="/dev/shm", type=str)  

    # parser.add_argument("--log_base_dir", default="/root/paddlejob/workspace/env_run/zhouzhen05/Data_SSD3/runs_hard_7B", type=str)
    parser.add_argument("--log_base_dir", default="runs_base_new_hard_composite", type=str)

    parser.add_argument("--exp_name", default="lisa-7b", type=str)
    parser.add_argument("--epochs", default=10, type=int)  # 10
    parser.add_argument("--steps_per_epoch", default=2300, type=int)  # The number of samples for each epoch：steps_per_epoch * batch_size * grad_accumulation_steps * num_gpu
    parser.add_argument(
        "--batch_size", default=16, type=int, help="batch size per device per step"   
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,  # 10
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=16, type=int)  
    parser.add_argument("--lr", default=0.0003, type=float)  

    parser.add_argument("--ce_loss_weight", default=1.0, type=float)  
    parser.add_argument("--dice_loss_weight", default=0.5, type=float) 
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)   

    parser.add_argument("--lora_alpha", default=16, type=int) 
    parser.add_argument("--lora_dropout", default=0.05, type=float)  
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str) 
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=1, type=int)  # For each question, repeat the num_classes_per_sample of sentences with the same meaning
    parser.add_argument("--exclude_val", action="store_true", default=False)

    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--vision_pretrained", default="checkpoint/sam", type=str)

    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create model
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        # "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    print("args.version: ", args.version, '*' * 100)
    # time.sleep(10)
    model = LISAForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "text_hidden_fcs",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make text_hidden_fcs(MLLM to seg), mask_decoder(seg), lm_head(detokenizer MLLM), embed_tokens(tokenizer MLLM) trainable
    if lora_r > 0:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]  # train MLLM and seg
                    # for x in ["lm_head", "embed_tokens"]  # only train MLLM 
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    else:
        for n, p in model.named_parameters():
            if any(
                [
                    x in n
                    # for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]  # train MLLM and seg
                    for x in ["mask_decoder", "text_hidden_fcs"]  # only train seg 
                ]
            ):
                # print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

            else:  # 若使用lora，把这里注释掉
                p.requires_grad = False

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, " ", p.requires_grad)

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        args.dataset_dir,
        tokenizer,
        args.vision_tower,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,  # The total amount of data to be sampled in one epoch is equal to len(train_dataset).
        precision=args.precision,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        exclude_val=args.exclude_val,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        explanatory=args.explanatory,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."  
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "data_loader": {
            "type": "pytorch",
            "pin_memory": True,  
            "num_workers": 16,   
            "persistent_workers": True  
        },
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    # train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1) 
    # import pdb;pdb.set_trace()
    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        N_acc, giou, ciou, bleu_score = validate(val_loader, model_engine, tokenizer, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        # print("first save", "*" * 100)
        # first_save_dir = os.path.join(args.log_dir, "first_ckpt_model")
        # if os.path.exists(first_save_dir):
        #     shutil.rmtree(first_save_dir)
        # model_engine.save_checkpoint(first_save_dir)
        # print("first save done", "*" * 100)

        if args.no_eval == False:
            N_acc, giou, ciou, bleu_score = validate(val_loader, model_engine, tokenizer, 0, writer, args)
            is_best = giou > best_score
            best_score = max(giou, best_score)
            cur_ciou = ciou if is_best else cur_ciou

        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


# def validate(val_loader, model_engine, epoch, writer, args):
#     intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
#     union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
#     acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

#     model_engine.eval()

#     for input_dict in tqdm.tqdm(val_loader):
#         torch.cuda.empty_cache()

#         input_dict = dict_to_cuda(input_dict)
#         if args.precision == "fp16":
#             input_dict["images"] = input_dict["images"].half()
#             input_dict["images_clip"] = input_dict["images_clip"].half()
#         elif args.precision == "bf16":
#             input_dict["images"] = input_dict["images"].bfloat16()
#             input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
#         else:
#             input_dict["images"] = input_dict["images"].float()
#             input_dict["images_clip"] = input_dict["images_clip"].float()

#         with torch.no_grad():
#             output_dict = model_engine(**input_dict)

#         pred_masks = output_dict["pred_masks"]
#         masks_list = output_dict["gt_masks"][0].int()
#         output_list = (pred_masks[0] > 0).int()
#         assert len(pred_masks) == 1

#         if len(masks_list) > 0:
#             intersection, union, acc_iou = 0.0, 0.0, 0.0
#             for mask_i, output_i in zip(masks_list, output_list):
#                 intersection_i, union_i, _ = intersectionAndUnionGPU(
#                     output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
#                 )
#                 intersection += intersection_i
#                 union += union_i
#                 acc_iou += intersection_i / (union_i + 1e-5)
#                 acc_iou[union_i == 0] += 1.0  # no-object target
#             intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
#             acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
#             intersection_meter.update(intersection), union_meter.update(union), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

#     intersection_meter.all_reduce()
#     union_meter.all_reduce()
#     acc_iou_meter.all_reduce()

#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
#     ciou = iou_class[1]
#     giou = acc_iou_meter.avg[1]

#     if args.local_rank == 0:
#         writer.add_scalar("val/giou", giou, epoch)
#         writer.add_scalar("val/ciou", ciou, epoch)
#         print("giou: {:.4f}, ciou: {:.4f}".format(giou, ciou))

#     # LISA-7B-v1-explanatory: giou: 0.6485, ciou: 0.6243
#     # LISA-7B-v1-explanatory-finetune: giou: 0.9474, ciou: 0.9545

#     # LISA-7B-v1-explanatory (1 to n): giou: 0.1646, ciou: 0.1076
#     # LISA-7B-v1-explanatory-finetune (1 to n): giou: 0.880, ciou: 0.903

#     return giou, ciou

def validate(val_loader, model_engine, tokenizer, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
    nt_tp_meter = AverageMeter("NT_TP", ":6.3f", Summary.SUM)
    nt_fn_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)
    bleu_meter = AverageMeter("NT_FN", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()
        device = input_dict["images"].device

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        "**********************************************   LM metric  **********************************************"
        input_ids = output_dict["input_ids"]
        output_ids = output_dict["output_empty_ids"]
        de_token_input = tokenizer.decode(input_ids, skip_special_tokens=False)
        de_token_output = tokenizer.decode(output_ids, skip_special_tokens=False)

        de_token_input = de_token_input.split('ASSISTANT: ')[1].split("</s>")[0]
        de_token_output = de_token_output.split('ASSISTANT: ')[1].split("</s>")[0]

        bleu_score = compute_bleu_with_dynamic_ngram([de_token_output], [de_token_input])
        bleu_meter.update(bleu_score)

        "**********************************************   reason_seg metric  **********************************************"
        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()
        assert len(pred_masks) == 1

        has_seg_empty = output_dict["has_seg_empty"]
        if masks_list.shape[0] == 0:
            if has_seg_empty == False:
                nt_tp_meter.update(1.0)
            else:
                nt_fn_meter.update(1.0)
                
        else:
            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                acc_iou += intersection_i / (union_i + 1e-5)
                acc_iou[union_i == 0] += 1.0  # no-object target
            
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]  
            intersection_meter.update(intersection), union_meter.update(union), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    nt_tp_meter.all_reduce()
    nt_fn_meter.all_reduce()
    bleu_meter.all_reduce()

    N_acc = nt_tp_meter.sum / (nt_tp_meter.sum + nt_fn_meter.sum)
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    bleu_score = bleu_meter.avg

    if args.local_rank == 0:
        writer.add_scalar("val/N_acc", N_acc, epoch)  # 97.5%
        writer.add_scalar("val/giou", giou, epoch)  # 63.6%
        writer.add_scalar("val/ciou", ciou, epoch)  # 64.3%
        writer.add_scalar("val/bleu", bleu_score, epoch)  # 91.7%
        print("N_acc: {:.4f}, giou: {:.4f}, ciou: {:.4f}, bleu: {:.4f}".format(N_acc, giou, ciou, bleu_score))

    return N_acc, giou, ciou, bleu_score


def compute_bleu_with_dynamic_ngram(pred, gt):
    pred_tokens = pred[0].split()
    gt_tokens = gt[0].split()
    
    # Dynamically adjust the maximum value of n-gram according to the text length
    max_n = min(len(pred_tokens), len(gt_tokens), 3)  # It supports up to 3-gram at most
    weights = tuple([1/max_n] * max_n + [0] * (4 - max_n))  # Dynamic allocation of weights
    
    smoothing_function = SmoothingFunction().method1  
    return sentence_bleu([gt_tokens], pred_tokens, weights=weights, smoothing_function=smoothing_function)


if __name__ == "__main__":
    # os.environ['HTTP_PROXY'] = 'agent.baidu.com:8188'
    # os.environ['HTTPS_PROXY'] = 'agent.baidu.com:8188'

    os.environ['MASTER_PORT'] = '24999'

    main(sys.argv[1:])
