import os
from datas.benchmark import Benchmark
from datas.dataset import DATASET
# from datas.df2k_realesrgan import DF2K_RealESRGAN
# from datas.df2k_denoise import DF2K_denoise
from torch.utils.data import DataLoader

def create_datasets(args):
    # create training dataset
    data = DATASET(
    args.training_data_path[0],
    args.training_data_path[1],
    train=True, 
    train_for_EA = False,
    augment=args.data_augment, 
    scale=args.scale, 
    colors=args.colors, 
    patch_size=args.patch_size, 
    repeat=args.data_repeat, 
    )
    train_dataloader = DataLoader(dataset=data, num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    data = DATASET(
    args.training_data_path[0],
    args.training_data_path[1],
    train=True, 
    train_for_EA = True,
    augment=args.data_augment, 
    scale=args.scale, 
    colors=args.colors, 
    patch_size=args.patch_size, 
    repeat=args.data_repeat, 
    )
    train_for_EA_dataloader = DataLoader(dataset=data, num_workers=args.threads, batch_size= args.batch_size_EA, shuffle=True)

    test_dataloaders = []
    for ind in range(len(args.eval_sets['names'])):
        name = args.eval_sets['names'][ind]
        hr_path = args.eval_sets['HR_paths'][ind]
        lr_path = args.eval_sets['LR_paths'][ind]
        val_data  = Benchmark(hr_path, lr_path, scale=args.scale, colors=args.colors)
        test_dataloaders += [{'name': name, 'dataloader': DataLoader(dataset=val_data, batch_size=1, shuffle=False)}]
    
    if len(test_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(0, len(test_dataloaders)):
            selected += test_dataloaders[i]['name'] + ", "
        print('select {} for evaluation! '.format(selected))
    
    fusion_dataloaders = []
    for ind in range(len(args.fusion_sets['names'])):
        name = args.fusion_sets['names'][ind]
        hr_path = args.fusion_sets['HR_paths'][ind]
        lr_path = args.fusion_sets['LR_paths'][ind]
        val_data  = Benchmark(hr_path, lr_path, scale=args.scale, colors=args.colors)
        fusion_dataloaders += [{'name': name, 'dataloader': DataLoader(dataset=val_data, batch_size=1, shuffle=False)}]
    
    if len(fusion_dataloaders) == 0:
        print('select no dataset for evaluation!')
    else:
        selected = ''
        for i in range(0, len(fusion_dataloaders)):
            selected += fusion_dataloaders[i]['name'] + ", "
        print('select {} for evaluation! '.format(selected))
        
    return train_dataloader, train_for_EA_dataloader, test_dataloaders, fusion_dataloaders