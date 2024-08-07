import os
import time
import random

import timm
import torch
import albumentations as A
import pandas as pd
import numpy as np
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo
import wandb

def wandb_login_init(train_time=None):
    load_dotenv()
    api_key = os.getenv('WANDB_API_KEY')
    wandb.login(key=api_key)

    if train_time is None:
        train_time = datetime.fromtimestamp(time.time(), tz=ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M%S")
        
    wandb.init(project="competition2-cv", name=f"run-{train_time}")
    
    print(f'train_time = {train_time}')

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# 데이터셋 클래스를 정의합니다.
class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None, aug_transform=None, augment_ratio=1):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform
        self.aug_transform = aug_transform
        self.augment_ratio = augment_ratio

    def __len__(self):
        return len(self.df) * self.augment_ratio

    def __getitem__(self, idx):
        real_idx = idx % len(self.df)
        name, target = self.df[real_idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        
        if idx >= len(self.df):
            assert self.aug_transform != None
            img = self.aug_transform(image=img)['image']
            
        elif self.transform:
            img = self.transform(image=img)['image']
            
        return img, target

# one epoch 학습을 위한 함수입니다.
def train_one_epoch(seed, loader, model, optimizer, loss_fn, device):
    # seed 를 다시 고정해서 동일한 augmentation 된 이미지로 학습 될 수 있도록 해보자.
    set_seed(seed)
    
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []

    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)

        model.zero_grad(set_to_none=True)

        preds = model(image)
        loss = loss_fn(preds, targets)
        
        #print(f"------ preds's shape = {preds.shape},  targets's shape = {targets.shape}")
        #print(f"---------- preds = {preds},  targets = {targets}")
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())

        pbar.set_description(f"Loss: {loss.item():.4f}")

    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')

    ret = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }
    
    # wandb에 학습 과정 로그
    wandb.log(ret)

    return ret

def create_trn_transform(trn_img_size):
    # augmentation을 위한 transform 코드
    trn_transform = A.Compose([
        # 이미지 크기 조정
        A.Resize(height=trn_img_size, width=trn_img_size),
        # images normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # numpy 이미지나 PIL 이미지를 PyTorch 텐서로 변환
        ToTensorV2(),
    ])
    return trn_transform

def create_trn_aug_transform(trn_img_size):
    trn_aug_transform = A.Compose([
        A.Resize(height=trn_img_size, width=trn_img_size),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 800.0), p=0.75),
            A.GaussianBlur(blur_limit=(1, 7), p=0.5)
        ], p=0.75),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.75),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.25),
        A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=0.5),
        A.ElasticTransform(alpha=1, sigma=30, alpha_affine=30, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Rotate(limit=30, p=0.75),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
        A.MotionBlur(blur_limit=5, p=0.5),
        A.OpticalDistortion(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return trn_aug_transform

# aug_p = 0.1
# trn_aug_transform = A.Compose([
#     A.Resize(height=img_size, width=img_size),
#     A.HorizontalFlip(p=aug_p),
#     A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=aug_p),
#     A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=aug_p),
#     A.CoarseDropout(max_holes=6, max_height=32, max_width=32, p=aug_p),
#     A.ElasticTransform(alpha=1, sigma=30, alpha_affine=30, p=aug_p),
#     A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=aug_p),
#     A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=aug_p),
#     A.Rotate(limit=(0, 360), p=1),
#     A.InvertImg(p=aug_p),
#     A.Solarize(threshold=128, p=aug_p),
#     #A.RandomCrop(height=img_size * 0.5, width=img_size * 0.5, p=aug_p),
#     A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=aug_p),
#     A.RandomGamma(gamma_limit=(80, 120), p=aug_p),
#     A.Posterize(num_bits=4, p=aug_p),
#     A.Equalize(p=aug_p),
#     A.GridDistortion(p=aug_p),
#     A.PiecewiseAffine(p=aug_p),
#     A.RandomShadow(p=aug_p),
#     A.RandomRain(p=aug_p),
#     A.RandomFog(p=aug_p),
#     A.RandomSunFlare(p=aug_p),
#     A.RandomSnow(p=aug_p),    
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2(),
# ])

def create_tst_transform(tst_img_size):
    # test image 변환을 위한 transform 코드
    tst_transform = A.Compose([
        A.Resize(height=tst_img_size, width=tst_img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return tst_transform

def train_dataset_split(train_csv_path, img_dir, trn_transform, trn_aug_transform, tst_transform, augment_ratio=1, train_size=0.7, random_state=42):
    # CSV 파일 읽기
    train_df = pd.read_csv(train_csv_path)
    
    train_df_target = train_df['target']
    train_df = train_df['ID']
    
    # 훈련 세트와 검증 세트로 분할
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_df, 
        train_df_target, 
        train_size=train_size, 
        stratify=train_df_target,
        random_state=random_state
    )
    
    print(f"[dataset_split] 훈련 세트: {len(X_train)} 샘플")
    print(f"[dataset_split] 검증 세트: {len(X_valid)} 샘플")
    
    train_df = pd.DataFrame({'ID':X_train, 'target':y_train})
    val_df = pd.DataFrame({'ID':X_valid, 'target':y_valid})
    
    # 각 데이터프레임을 임시 CSV 파일로 저장
    train_df.to_csv('temp_train.csv', index=False)
    val_df.to_csv('temp_val.csv', index=False)
    
    # ImageDataset 생성
    train_dataset = ImageDataset(
        'temp_train.csv', 
        img_dir, 
        transform=trn_transform, 
        aug_transform=trn_aug_transform, 
        augment_ratio=augment_ratio)
    
    val_dataset = ImageDataset(
        'temp_val.csv', 
        img_dir, 
        transform=tst_transform)
    
    # 임시 파일 삭제
    os.remove('temp_train.csv')
    os.remove('temp_val.csv')
    
    return train_dataset, val_dataset

def train_with_start_end_epoch(seed,
                               tst_img_size, 
                               batch_size, 
                               start_epoch_inclusive, 
                               end_epoch_exclusive, 
                               augment_ratio,
                               trn_loader,
                               val_loader,
                               model,
                               model_name,
                               optimizer,
                               loss_fn,
                               device,
                               is_save_model_checkpoint,
                               is_evaluate_train_valid,
                               fold = 0,
                               folds = 0):
    
    for epoch in range(start_epoch_inclusive, end_epoch_exclusive):
        epoch += 1
        
        print("\n=================================================================")
        ret = train_one_epoch(seed, trn_loader, model, optimizer, loss_fn, device=device)
        ret['epoch'] = epoch

        log = ""
        for k, v in ret.items():
            log += f"{k}: {v:.4f}\n"
        print(log)
        
        eval_str = ""

        if is_evaluate_train_valid and (val_loader != None):
            evalDict = evaluate_train_valid(seed, None, val_loader, model, loss_fn, device)
            
            if 'valid_results' in evalDict:
                valid_results = evalDict['valid_results']            
                eval_str = f'vl_{valid_results[0]:.4f}_va_{valid_results[1]:.4f}_vf1_{valid_results[2]:.4f}'
        
        if is_save_model_checkpoint:
            save_model_checkpoint(seed, tst_img_size, batch_size, epoch, augment_ratio, model, model_name, optimizer, eval_str, fold, folds)

def count_error_preds(preds, targets):
    error_counts = {}

    for pred, target in zip(preds, targets):
        if pred == target: continue
        
        if target not in error_counts:
            error_counts[target] = 1
        else:
            error_counts[target] += 1
            
    return f"{sum(error_counts.values())}/{len(targets)}, {error_counts}"

def evaluate(seed, loader, model, loss_fn, device):
    # seed 를 다시 고정해서 학습할때 사용했던 augmentation 이 사용될 수 있도록 수정.
    set_seed(seed)

    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for image, targets in tqdm(loader, desc="Evaluating"):
            image = image.to(device)
            targets = targets.to(device)

            preds = model(image)
            loss = loss_fn(preds, targets)

            total_loss += loss.item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')

    # wandb에 평가 메트릭 로깅
    results = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1": f1
    }
    #wandb.log(results)

    return avg_loss, accuracy, f1, count_error_preds(all_preds, all_targets)

def evaluate_train_valid(seed, trn_loader, val_loader, model, loss_fn, device):
    if trn_loader != None:
        train_results = evaluate(seed, trn_loader, model, loss_fn, device)
    
    if val_loader != None:
        valid_results = evaluate(seed, val_loader, model, loss_fn, device)

    # 평가 결과 로깅
    log_dict = {}
    
    if trn_loader != None:
        log_dict["final_train_loss"] = train_results[0]
        log_dict["final_train_accuracy"] = train_results[1]
        log_dict["final_train_f1"] = train_results[2]
    
    if val_loader != None:
        log_dict["final_valid_loss"] = valid_results[0]
        log_dict["final_valid_accuracy"] = valid_results[1]
        log_dict["final_valid_f1"] = valid_results[2]

    wandb.log(log_dict)

    print()
    for k, v in log_dict.items():
        print(f'{k}: {v}')
    
    if trn_loader != None:
        print(f"train's error preds count: {train_results[3]}")
    
    if val_loader != None:
        print(f"valid's error preds count: {valid_results[3]}")
    
    retDic = {}
    if trn_loader != None:
        retDic['train_results'] = train_results
    
    if val_loader != None:
        retDic['valid_results'] = valid_results
    
    return retDic

def save_model_checkpoint(seed, tst_img_size, batch_size, epoch, augment_ratio, model, model_name, optimizer, postfix, fold = 0, folds = 0):
    cp_filename = f"cp-{model_name}_sd_{seed}_epc_{epoch}_aug_{augment_ratio}"
    
    if len(postfix) > 0:
        cp_filename += f"_{postfix}"
    
    if folds > 0:
        cp_filename += f"_fold_{fold}_folds_{folds}"
    
    cp_filename += ".pt"
    
    torch.save(
        {
            "model": model_name,
            "seed": seed,
            "tst_img_size": tst_img_size,
            "batch_size": batch_size,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "description": f"{model_name} 모델 epoch {epoch} 까지 학습한 모델, fold: {fold}/{folds}",
        },
        cp_filename
    )
    
    print(f"Model checkpoint saved. filename: {cp_filename}")

def load_model_checkpoint(cp_filename, model, optimizer, device):
    checkpoint = torch.load(cp_filename, map_location = device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint

def get_preds_list_by_tst_loader(model, tst_loader, device, is_soft_voting=False):
    preds_list = []

    model.eval()
    for image, _ in tqdm(tst_loader):
        image = image.to(device)

        with torch.no_grad():
            preds = model(image)
            
        if is_soft_voting:
            soft_preds = F.softmax(preds, dim=1)
            preds_list.extend(soft_preds.detach().cpu().numpy())
        else:
            preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
    
    return preds_list

def pred_and_save_to_csv(model, tst_loader, device, csv_filename):
    preds_list = get_preds_list_by_tst_loader(model, tst_loader, device)
    preds_list_to_save_to_csv(preds_list, tst_loader, csv_filename)

def preds_list_to_save_to_csv(preds_list, tst_loader, csv_filename):
    pred_df = pd.DataFrame(tst_loader.dataset.df, columns=['ID', 'target'])
    pred_df['target'] = preds_list

    sample_submission_df = pd.read_csv("datasets_fin/sample_submission.csv")
    assert (sample_submission_df['ID'] == pred_df['ID']).all()

    pred_df.to_csv(csv_filename, index=False)

def get_fold_train_valid_csv_filenames(seed, fold, folds):
    fold_train_filename = f'fold_train_SEED_{seed}_{fold}_{folds}.csv'
    fold_valid_filename = f'fold_valid_SEED_{seed}_{fold}_{folds}.csv'
    return fold_train_filename, fold_valid_filename

def generate_fold_train_valid_csv_files(seed, folds):
    df_train = pd.read_csv("datasets_fin/train.csv")
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    for fold, (train_indices, valid_indices) in enumerate(skf.split(df_train, df_train['target'])):
        fold += 1
        
        print(f"Fold {fold}/{folds}, train_idx: {type(train_indices)} {len(train_indices)}, {type(valid_indices)} {len(valid_indices)}")
        #print(f"     train_indices: {train_indices}, valid_indices: {valid_indices}")
        #print(f'train rows: {df_train.iloc[train_indices]}')
        #print(f'valid rows: {df_train.iloc[valid_indices]}')
        
        fold_train_filename, fold_valid_filename = get_fold_train_valid_csv_filenames(seed, fold, folds)
        
        # 각 데이터프레임을 임시 CSV 파일로 저장
        df_train.iloc[train_indices].to_csv(fold_train_filename, index=False)
        df_train.iloc[valid_indices].to_csv(fold_valid_filename, index=False)

def get_supplies_for_train_and_valid_with_fold(seed, 
                                               model_name, 
                                               lr,
                                               batch_size, 
                                               num_workers, 
                                               fold,
                                               folds, 
                                               augment_ratio, 
                                               trn_img_size, 
                                               tst_img_size, 
                                               device):
    
    fold_train_filename, fold_valid_filename = get_fold_train_valid_csv_filenames(seed, fold, folds)
    
    trn_transform = create_trn_transform(trn_img_size)
    trn_aug_transform = create_trn_aug_transform(trn_img_size)
    tst_transform = create_tst_transform(tst_img_size)
    
    # ImageDataset 생성
    trn_dataset = ImageDataset(
        fold_train_filename, 
        "datasets_fin/train/",
        transform=trn_transform, 
        aug_transform=trn_aug_transform, 
        augment_ratio=augment_ratio)
    
    val_dataset = ImageDataset(
        fold_valid_filename, 
        "datasets_fin/train/",
        transform=tst_transform)
    
    # DataLoader 정의
    trn_loader = DataLoader(
        trn_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True,
        drop_last = False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size = batch_size,
        shuffle = False,
        num_workers = 0,
        pin_memory = True
    )
    
    # load model
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=17
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    
    retDict = {
        "trn_loader": trn_loader,
        "val_loader": val_loader,
        "model": model,
        "optimizer": optimizer,
        "loss_fn": loss_fn,
    }
    
    return retDict
