import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

# 1. Configuration
class CFG:
    epochs = 50
    batch_size = 32  
    num_workers = 4 

    # Learning Rates
    head_lr = 1e-6           
    image_encoder_lr = 5e-6  
    text_encoder_lr = 1e-6 
    
    weight_decay = 0.1
    projection_dim = 256

    
    model_text = "distilbert-base-uncased"

    resnet_weights = "./out/Resnet/resnet18_supervised_run3.pt"
    resnet_checkpoints = os.path.join("./out/Resnet/", "resnet18_clip_run3.pt")
    out_dir = "./out/Resnet/"


# 2. Dataset
class CIFAR10H_CLIP_Dataset(torch.utils.data.Dataset):
    def __init__(self, cifar_dataset, captions_dict, tokenizer, transforms=None, is_train=True):
        self.cifar_dataset = cifar_dataset  
        self.captions = captions_dict 
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.is_train = is_train
        self.indices = list(captions_dict.keys())
        self.to_pil = torchvision.transforms.ToPILImage()

    def __getitem__(self, idx):
        key = self.indices[idx]
        img, label_idx = self.cifar_dataset[int(key)]

        caption_list = self.captions[key]["captions"]        

        if self.is_train:
            # Choose one of the three captions for training
            caption = random.choice(caption_list)
        else:
            # Take one of the captions for consistency during validation
            caption = caption_list[0]
            
        if isinstance(img, torch.Tensor):
            img = self.to_pil(torch.clamp(img, 0, 1))
            img = self.to_pil(img)

        if self.transforms:
            img = self.transforms(img)

        tokenized = self.tokenizer(
            caption, padding='max_length', max_length=77, 
            truncation=True, return_tensors="pt"
        ) 

        return {
            'image': img,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
            'label': label_idx,
            'caption': caption
        }

    def __len__(self):
        return len(self.indices)

# --- 3. Models ---
class ResImageEncoder(nn.Module):
    def __init__(self, checkpoint_path=None):
        super().__init__()
        # 1. load basic modell
        self.model = models.resnet18(weights=None)
        
        # 2. Reduce classifier head to 10 dimensions
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
        
        # 3. load weights
        if checkpoint_path:
            print(f">>> Loading Resnet weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
        
        # 4. prepare feature extraction
        self.embedding_dim = 512  # ResNet18/34 use BasicBlock (expansion=1) -> 512, not 2048 like ResNet50+
        self.model.fc = nn.Identity() # get rid of head

    def forward(self, x):
        # making sure every image is 224 X 224
        if x.shape[-1] != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)



class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(model_name)
        self.embedding_dim = 768

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :] 



class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        # increasing the embedding dimension for more capacity
        intermediate_dim = embedding_dim * 2 
        self.fc1 = nn.Linear(embedding_dim, intermediate_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(intermediate_dim, projection_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(projection_dim)

        # Shortcut für den Residual-Pfad
        self.shortcut = nn.Linear(embedding_dim, projection_dim)



    def forward(self, x):
        residual = self.shortcut(x)
        # Expansion -> Activation -> Reduction
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.drop(self.fc2(x))
        # Residual Connection + LayerNorm
        return self.ln(x + residual)


class CLIPNet(nn.Module):
    def __init__(self, text_model_name, projection_dim):
        super().__init__()
        self.image_encoder = ResImageEncoder(checkpoint_path=CFG.vit_weights)
        self.text_encoder = TextEncoder(text_model_name)

        
        self.image_projection = ProjectionHead(self.image_encoder.embedding_dim, projection_dim)
        self.text_projection = ProjectionHead(768, projection_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.classifier = nn.Linear(projection_dim, 10)



    def forward(self, batch):

        img_feats = self.image_encoder(batch["image"])
        txt_feats = self.text_encoder(batch["input_ids"], batch["attention_mask"])
        
        img_emb = F.normalize(self.image_projection(img_feats), dim=-1, eps=1e-6)
        txt_emb = F.normalize(self.text_projection(txt_feats), dim=-1, eps=1e-6)

        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)
            
        logit_scale = self.logit_scale.exp()
        logits = (img_emb @ txt_emb.T) * logit_scale

        

        # 4. CLIP Loss (Kontrastiv)
        ground_truth = torch.arange(len(logits), device=logits.device)
        loss = (F.cross_entropy(logits, ground_truth) + 
                F.cross_entropy(logits.T, ground_truth)) / 2
        
        return loss

def train_epoch(model, loader, optimizer, scheduler, device, scaler, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}", leave=False)

    for batch in pbar:
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            loss = model(batch)

        if scaler is not None:
            scaler.scale(loss).backward()
        
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.8f}")

    return total_loss / len(loader)
    
def validate(model, loader, device, tokenizer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Pre-compute text features for accuracy once per validation call
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    prompts = [f"a photo of a {c}" for c in class_names]

    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)

    with torch.no_grad():
        text_feats = F.normalize(model.text_projection(model.text_encoder(inputs['input_ids'], inputs['attention_mask'])), p=2, dim=-1)
        
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.amp.autocast('cuda'):
                # calling the forward function to compute loss
                loss = model(batch)

                # using the image features for the accuracy
                img_feats = F.normalize(model.image_projection(model.image_encoder(batch["image"])), p=2, dim=-1)
                logits = (img_feats @ text_feats.T)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch["label"]).sum().item()
                total += batch["label"].size(0)

            total_loss += loss.item()            

    return total_loss / len(loader), correct / total



def calculate_class_accuracy(model, loader, device, tokenizer):
    model.eval()
    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    prompts = [f"a photo of a {c}" for c in class_names]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
    correct, total = 0, 0

    with torch.no_grad():
        text_features = model.text_encoder(inputs['input_ids'], inputs['attention_mask'])
        text_features = model.text_projection(text_features)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=-1)

        for batch in loader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            image_features = model.image_encoder(images)
            image_features = model.image_projection(image_features)
            image_features = torch.nn.functional.normalize(image_features, p=2, dim=-1)
            logits = (image_features @ text_features.T)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total if total > 0 else 0



# 4. LLRD Optimizer Setup
def get_llrd_optimizer_params(model, base_lr, head_lr, decay):
    """
    Optimized for ResNet18 + DistilBERT CLIP.
    makes sure, that gradients dont explode or vanish
    """
    param_groups = [
        # Die Projektionsköpfe sind am wichtigsten für die Ausrichtung
        {"params": model.image_projection.parameters(), "lr": head_lr},
        {"params": model.text_projection.parameters(), "lr": head_lr},
        # Logit Scale separat mit niedriger LR, um NaNs zu vermeiden
        {"params": [model.logit_scale], "lr": 1e-4} 
    ]

    # nackwards path through resnet
    # Layer 4 get full base lr
    layers = [
        model.image_encoder.model.layer4,
        model.image_encoder.model.layer3,
        model.image_encoder.model.layer2,
        model.image_encoder.model.layer1,
        nn.Sequential(model.image_encoder.model.conv1, model.image_encoder.model.bn1)
    ]
    
    current_lr = base_lr
    for layer in layers:
        param_groups.append({"params": layer.parameters(), "lr": current_lr})
        # decay of 0.8-0.9 is ideal for small lr
        current_lr *= decay 

    # DistilBERT: takes consistenly small lr
    param_groups.append({"params": model.text_encoder.parameters(), "lr": base_lr})
    
    return param_groups

# 5. Main Execution
if __name__ == "__main__":

    res_standard_pretrained_path = CFG.resnet_weights

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.out_dir, exist_ok=True)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.model_text)

    # Dataset & Loader
    mean= [0.489, 0.4321, 0.4976]
    std = [0.22734, 0.24983, 0.2357]
    
    def get_training_transforms(): 
        return T.Compose([
            T.Resize((224, 224),  interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    
    def get_validation_transforms():
        return T.Compose([
            T.Resize((224, 224),  interpolation=T.InterpolationMode.BICUBIC  ),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
    
    train_cifar = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    test_cifar = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    
    with open("cifar10_train_captions.json", "r") as f: train_captions = json.load(f)
    with open("cifar10_test_captions.json", "r") as f: test_captions = json.load(f)

    # Dataset Setup 
    # Create datasets 
    train_ds = CIFAR10H_CLIP_Dataset(
        train_cifar,
        captions_dict=train_captions,
        tokenizer=tokenizer,
        transforms=get_training_transforms(),
        is_train=True
    )
    
    val_ds = CIFAR10H_CLIP_Dataset(
        test_cifar,
        captions_dict=test_captions,
        tokenizer=tokenizer,
        transforms=get_validation_transforms(),
        is_train=False
    )
    
    
    # DataLoaders
    
    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers
    )

        
    model = CLIPNet(
        text_model_name=CFG.model_text,
        projection_dim=CFG.projection_dim
    ).to(device)

    if os.path.exists(res_standard_pretrained_path):
        print(f">>> Initialisiere Resnet-Backbone with weights from: {res_standard_pretrained_path}")
            
    optimizer_params = get_llrd_optimizer_params(model, CFG.head_lr, CFG.image_encoder_lr, decay=0.85)
    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=CFG.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[p['lr'] * 5 for p in optimizer_params], 
        steps_per_epoch=len(train_loader), 
        epochs=CFG.epochs
    )
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    for epoch in range(CFG.epochs):
        # Freeze Image Encoder for the first 3 Epochs
        for param in model.image_encoder.parameters():
            param.requires_grad = (epoch >= 3)
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, scaler, epoch)
        _, val_acc = validate(model, val_loader, device, tokenizer)
        
        print(f"Summary | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CFG.r_checkpoints)
            print(f">>> New best model saved ({best_acc*100:.2f}%)")