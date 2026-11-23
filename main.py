import os
import timm
import torch
import argparse
from tqdm import tqdm
from src.models import get_models, get_encoder_decoders, get_ranks
from src.utils import (seed_everything, get_dataloaders_imagenet, get_dataloaders_cifar10, get_layer_names, create_optimizer_scheduler, 
                       process_model_blocks, analyze_parameter_storage)
from src.training import initial_fit, train


def main(args):
    seed_everything()
    os.makedirs(f'saved_models/{args.base_dir}', exist_ok=True)
    torch.cuda.set_device(args.device)

    args.hidden_dim = 768 if 'base' in args.model_name else 384
    print('hidden_dim:', args.hidden_dim)
    print('ranks per layer type:', get_ranks(args))

    # Load and prepare the model
    model, compressed_model = get_models(args)

    selected_layers = get_layer_names(args)

    # extract original weights and input sequence lengths (batch size of weights)
    original_weights, input_seq_lens = process_model_blocks(model, compressed_model, args, selected_layers=selected_layers, 
                                                            skip_qkv=args.skip_qkv)

    # Get data loaders
    if args.dataset == 'cifar10':
        train_loader, val_loader = get_dataloaders_cifar10(args.batch_size, root=args.data_root, num_workers=args.num_workers)
    else:
        train_loader, val_loader = get_dataloaders_imagenet(args.batch_size)

    encoders, decoders = get_encoder_decoders(selected_layers, input_seq_lens, args)
    print('original_weights:', original_weights.keys())
    print('encoder:', encoders)
    print('decoder:', decoders)

    analyze_parameter_storage(compressed_model, encoders, decoders, original_weights, args)

    # Perform initial training steps to fit the decoder to the original weights
    initial_fit(encoders, decoders, original_weights, args)

    # Create optimizer and scheduler
    optimizer, scheduler, optimizer_ft, scheduler_ft = create_optimizer_scheduler(compressed_model, encoders, decoders, args)
        
    # distilled model for knowledge distillation
    # ImageNet: the teacher is the pretrained distilled DeiT (as in the paper).
    # CIFAR-10: the 1000-class distilled DeiT cannot teach a 10-class head, so the teacher is the dense
    # CIFAR-10 DeiT-S that get_models() already loaded into `model` (97.29%).
    if args.distilled_model and args.dataset == 'imagenet':
        del model
        torch.cuda.empty_cache()
        model = timm.create_model('deit_small_distilled_patch16_224', pretrained=True) if 'small' in args.model_name else timm.create_model('deit_base_distilled_patch16_224', pretrained=True)
        model.to(args.device)
        
    model.eval()  # the teacher is never trained

    # Train the model
    best_acc = train(encoders, decoders, original_weights, model, compressed_model, train_loader, val_loader, 
                     optimizer, scheduler, optimizer_ft, scheduler_ft, selected_layers, args)

    print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Vision Transformer with weight compression')
    parser.add_argument('--model_name', type=str, default='deit_small_patch16_224', help='Name of the Vision Transformer model')
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['cifar10', 'imagenet'], help='Dataset to use for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the Autoencoder')
    parser.add_argument('--min_lr', type=float, default=0, help='Min. Learning rate for the Autoencoder')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the Autoencoder')
    parser.add_argument('--total_blocks', type=int, default=12, help='Number of blocks to compress')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training')
    parser.add_argument('--initial_iters', type=int, default=1000, help='Number of initial iterations to train the Autoencoder')
    parser.add_argument('--eval_interval', type=int, default=200000, help='Number of iterations to evaluate the model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for ImageNet dataloader')
    parser.add_argument('--base_dir', type=str, default='test', help='Base directory for saving models')
    parser.add_argument('--skip_qkv', action='store_true', help='Skip compressing qkv layer in MultiheadAttention')
    parser.add_argument('--mixup', action='store_true', help='Use Mixup')
    parser.add_argument('--ce_weight', type=float, default=1.0, help='Weight for distillation loss')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='Weight for MSE loss')
    parser.add_argument('--distillation_weight', type=float, default=3e3, help='Weight for distillation loss')
    parser.add_argument('--finetune_other_params', action='store_true', help='Finetune other parameters of the model')
    parser.add_argument('--opt', type=str, default='adamw', help='Optimizer to use for finetuning other parameters')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--rank', type=int, default=276, help='Rank for encoded weight matrices (used for every layer type unless overridden below)')
    parser.add_argument('--rank_qkv', type=int, default=None, help='Rank for the qkv layers (overrides --rank)')
    parser.add_argument('--rank_proj', type=int, default=None, help='Rank for the proj layers (overrides --rank)')
    parser.add_argument('--rank_fc1', type=int, default=None, help='Rank for the fc1 layers (overrides --rank)')
    parser.add_argument('--rank_fc2', type=int, default=None, help='Rank for the fc2 layers (overrides --rank)')
    parser.add_argument('--data_root', type=str, default='./data/cifar10', help='Where CIFAR-10 is stored / downloaded')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers (CIFAR-10)')
    parser.add_argument('--time_iters', type=int, default=0, help='If > 0: time this many training iterations, print the estimated epoch time and exit')
    parser.add_argument('--distilled_model', action='store_true', help='Use distilled model for distillation')
    parser.add_argument('--warmup', action='store_true', help='Use warmup scheduler')
    args = parser.parse_args()
    args.finetune_other_params = True
    args.distilled_model = True  # on CIFAR-10 the dense CIFAR-10 model is used as teacher instead (see main())
    # args.warmup = True
    main(args)
