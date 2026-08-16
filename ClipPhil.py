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
from tqdm import tqdm
from transformers import DistilBertModel, DistilBertTokenizer

# 1. Configuration
class CFG:
    epochs = 50
    batch_size = 256  
    num_workers = 4 

    # Learning Rates
    head_lr = 1e-5           
    image_encoder_lr = 1e-6  
    text_encoder_lr = 1e-6 
    
    weight_decay = 0.1
    projection_dim = 256

    
    model_text = "distilbert-base-uncased"

    philnet_weights = "./out/Philnet/philnet_supervised_run1.pt"
    checkpoint_path = os.path.join("./out/Philnet/", "./philnet_clip_run1.pt")

    out_dir = "./out/Philnet/"


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
            # choose one of the 3 captions
            caption = random.choice(caption_list)
        else:
            # take one of the captions for validation
            caption = caption_list[0]
            
        if isinstance(img, torch.Tensor):
            img = self.to_pil(torch.clamp(img, 0, 1))

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

# 3. Models 
class PhilNetImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Philnet as image encoder
        self.model = nn.Sequential(
            
            # Block 1 (Input: 32x32)
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # Result: 16x16

            # Block 2 (16x16)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2),  # Result: 8x8

            # Block 3 (8x8)
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),  
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1) # Result: 512x1x1
        )

        

        # dimension for the projection head
        self.embedding_dim = 512
        print("PhilNet CNN Backbone ready for CLIP training.")


    def forward(self, x):
        # Forward pass through the Convolutional Layers
        x = self.model(x)
        # Flatten von (Batch, 512, 1, 1) auf (Batch, 512)
        return torch.flatten(x, 1)



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
        self.image_encoder = PhilNetImageEncoder()
        self.text_encoder = TextEncoder(text_model_name)

        
        self.image_projection = ProjectionHead(512, projection_dim)
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

        

        # 4. CLIP Loss (contrastive)
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



def get_llrd_optimizer_params(model, base_lr, decay_factor=0.85):
    params = []
    # Heads & logit_scale
    head_params = [p for n, p in model.named_parameters() if "encoder" not in n]
    params.append({'params': head_params, 'lr': base_lr})
    
    # Text-Encoder (BERT)
    params.append({'params': model.text_encoder.parameters(), 'lr': base_lr * 0.1})

    # PhilNet-Backbone
    layers = list(model.image_encoder.model.children())
    layers.reverse() 
    current_lr = base_lr * 0.5 

    for layer in layers:
        layer_p = [p for p in layer.parameters() if p.requires_grad]
        if len(layer_p) > 0:
            params.append({'params': layer_p, 'lr': current_lr})
            current_lr *= decay_factor # Jede tiefere Schicht lernt langsamer
            
    return params


# --- 5. Main Execution ---
if __name__ == "__main__":

    philnet_pretrained_path = "./out/Philnet/philnet_supervised_run1.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CFG.out_dir, exist_ok=True)
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.model_text)

    # Dataset & Loader
    mean= [0.489, 0.4321, 0.4976]
    std = [0.22734, 0.24983, 0.2357]

    def get_training_transforms(): 
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize(mean, std)
           ])
    
    
    
    def get_validation_transforms():
        return T.Compose([
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

    clip_model = CLIPNet(
        text_model_name=CFG.model_text,
        projection_dim=CFG.projection_dim
    ).to(device)

    if os.path.exists(philnet_pretrained_path):
    
        print(f">>> Loading PhilNet backbone from: {philnet_pretrained_path}")
    
        pretrained_state = torch.load(
            philnet_pretrained_path,
            map_location=device
        )
    
        # Handle different checkpoint formats
        if isinstance(pretrained_state, dict) and 'model_state_dict' in pretrained_state:
            pretrained_state = pretrained_state['model_state_dict']
    
        elif isinstance(pretrained_state, dict) and 'model' in pretrained_state:
            pretrained_state = pretrained_state['model']
    
    
        current_model_dict = clip_model.image_encoder.model.state_dict()
    
        new_state_dict = {}
    
        p_layers = list(pretrained_state.keys())
        c_layers = list(current_model_dict.keys())
    
        # Map PhilNet weights into CLIP image encoder
        for i in range(min(len(p_layers), len(c_layers))):
    
            if pretrained_state[p_layers[i]].shape == current_model_dict[c_layers[i]].shape:
                new_state_dict[c_layers[i]] = pretrained_state[p_layers[i]]
    
        if len(new_state_dict) > 0:
    
            clip_model.image_encoder.model.load_state_dict(
                new_state_dict,
                strict=False
            )
    
            print(f">>> SUCCESS: {len(new_state_dict)} PhilNet layers mapped!")
    
        else:
    
            print(">>> ERROR: No matching layer shapes found.")
        
    optimizer_params = get_llrd_optimizer_params(
        clip_model,
        base_lr=CFG.head_lr
    )
    
    optimizer = torch.optim.AdamW(
        optimizer_params,
        weight_decay=CFG.weight_decay
    )
    
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group['lr'] * 5 for group in optimizer.param_groups],
        steps_per_epoch=len(train_loader),
        epochs=CFG.epochs,
        pct_start=0.1,
        div_factor=10,
        final_div_factor=1000
    )

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    for epoch in range(CFG.epochs):
        # Freeze Image Encoder for first 10 epochs
        for param in clip_model.image_encoder.parameters():
            param.requires_grad = (epoch >= 10)
        
        train_loss = train_epoch(clip_model, train_loader, optimizer, scheduler, device, scaler, epoch)
        val_loss, val_acc = validate(clip_model, val_loader, device, tokenizer)
        
        print(f"Summary | Train Loss: {train_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(clip_model.state_dict(), CFG.checkpoint_path)
            print(f">>> New best model saved ({best_acc*100:.2f}%)")