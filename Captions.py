import os
import json
import torch
import random
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import numpy as np

def generate_dataset_captions(base_dataset, output_file, num_captions_per_image=3):
    captions_dict = {}
    
    # Checkpoint laden
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            captions_dict = json.load(f)
        print(f"Resuming from index {len(captions_dict)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()

    blip_transform = T.Compose([
        T.ToPILImage(),
        # ETry a good resize algorithm
        T.Lambda(lambda img: torch.from_numpy(
            cv2.resize(np.array(img), (224, 224), interpolation=cv2.INTER_LANCZOS4)
        ).permute(2, 0, 1).float() / 255.0),
    ])

    class_names = base_dataset.classes

    # labels
    class_names = ["airplane","automobile","bird","cat","deer",
                   "dog","frog","horse","ship","truck"]
    prompt_templates = [
        "a photo of a {label}",
        "an image of a {label}",
        "a picture of a {label}",
        "a cropped photo of a {label}",
        "a blurry photo of a {label}"
    ]

    # 1. prperation filtering of overused wrods
    forbidden_phrases = ["black background", "dark background", "black", "in the dark", "dark", "night", "with the words"]
    bad_words_ids = processor.tokenizer(forbidden_phrases, add_special_tokens=False).input_ids
    
    print("Generating enriched captions with BLIP...")    
    for i in tqdm(range(len(base_dataset))):
        if str(i) in captions_dict:
            continue
            
        img_tensor, label_idx = base_dataset[i]
        label_name = class_names[label_idx]
        
        raw_image = blip_transform(torch.clamp(img_tensor, 0, 1)).unsqueeze(0).to(device)
        current_captions = []
        
        with torch.no_grad():
            # random starting prompt
            base_prompt = random.choice(prompt_templates).format(label=label_name)
            
            inputs = processor(images=raw_image, text=base_prompt, return_tensors="pt").to(device)
            
            # Generation. good enough
            out = model.generate(
                **inputs,
                max_new_tokens=15,
                min_new_tokens=5,
                num_beams=12,
                num_beam_groups=3,
                num_return_sequences=3,
                diversity_penalty=8.0,
                repetition_penalty=5.0,      # strong penalty for word repetitions
                bad_words_ids=bad_words_ids, # forbiden tokens
                do_sample=False
            )
            
            # extract 3 different captions
            for j in range(len(out)):
                caption = processor.decode(out[j], skip_special_tokens=True).lower().strip()
                
                # save loaded prompts
                if label_name.lower() not in caption:
                    caption = f"{label_name} {caption}"
                
                current_captions.append(caption)

        # save loaded prompts
        captions_dict[str(i)] = {
            "label_id": int(label_idx),
            "label_name": label_name,
            "captions": current_captions[:3] # Sicherstellen, dass es genau 3 sind
        }

        # Auto-Save all 100 Pictures
        if i % 100 == 0:
            with open(output_file, "w") as f:
                json.dump(captions_dict, f, indent=4)

    # Final save
    with open(output_file, "w") as f:
        json.dump(captions_dict, f, indent=4)
    
    return captions_dict

# MAIN EXECUTION BLOCK
if __name__ == "__main__":
    transform = T.Compose([T.ToTensor()])
    train_dataset = torchvision.datasets.CIFAR10("./data/", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10("./data/", train=False, download=True, transform=transform)
    
    # 1. generate captions for TRAIN (50.000 pictures)
    train_captions = generate_dataset_captions(
        train_dataset, 
        output_file="cifar10_train_captions_5x.json", 
        num_captions_per_image=3
    )
    
    # 2. Generate for TEST (10.000 pictures)
    test_captions = generate_dataset_captions(
        test_dataset, 
        output_file="cifar10_test_captions_5x.json", 
        num_captions_per_image=3
    )