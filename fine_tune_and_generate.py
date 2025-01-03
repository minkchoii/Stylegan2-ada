import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from huggingface_hub import login
from diffusers import StableDiffusionPipeline
import glob

# 데이터 경로
dataset_path = '/C/User/chlalsrud/Desktop/Ship'
csv_path = '/C/User/chlalsrud/Desktop/image_hashes_with_prompts.csv'

# CSV 파일 로드
df = pd.read_csv(csv_path)

# 이미지-프롬프트 매핑
hash_to_prompt = {row['Hash Name']: row['Prompt'] for _, row in df.iterrows()}

# 데이터셋 정의
class CustomDataset(Dataset):
    def __init__(self, image_dir, csv_path, transform=None):
        self.image_dir = image_dir
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        hash_value = self.df.iloc[idx]['Hash Name']
        # 모든 하위 디렉토리에서 파일 탐색
        image_path = glob.glob(os.path.join(self.image_dir, f"**/{hash_value}.*"), recursive=True)
        if not image_path:
            raise FileNotFoundError(f"No file found for hash: {hash_value}")
        image = Image.open(image_path[0]).convert("RGB")
        prompt = self.df.iloc[idx]['Prompt']
        if self.transform:
            image = self.transform(image)
        return image, prompt

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 크기를 줄여 GPU 메모리 사용량 감소
    transforms.ToTensor(),
])

dataset = CustomDataset(dataset_path, csv_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)

# GPU 메모리 최적화
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# Hugging Face 로그인
login("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")

# Stable Diffusion 모델 로드
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-base",
    torch_dtype=torch.float16
)
pipe.to("cuda")  # GPU로 이동
pipe.unet = pipe.unet.half()  # UNet FP16 변환
pipe.vae = pipe.vae.half()  # VAE FP16 변환


# 텍스트 인코더는 FP32로 유지
pipe.text_encoder = pipe.text_encoder.to("cuda").float()

# 모델 미세 조정 (Fine-tuning) with Gradient Accumulation
def fine_tune_model(pipe, dataloader, num_epochs=1, accumulation_steps=4):
    pipe.unet.train()
    optimizer = torch.optim.Adam(pipe.unet.parameters(), lr=5e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = torch.amp.GradScaler()

    # 모델을 FP32로 설정
    pipe.unet = pipe.unet.to(torch.float32)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for step, (images, prompts) in enumerate(dataloader):
            images = images.to(device).to(torch.float32)  # FP32로 변환

            # 3채널(RGB)을 4채널로 변환
            batch_size, _, height, width = images.shape
            images_4channel = torch.cat([images, torch.zeros(batch_size, 1, height, width).to(device)], dim=1)

            # 텍스트 임베딩 생성
            text_inputs = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            text_embeddings = pipe.text_encoder(text_inputs)[0]

            # 타임스텝 생성
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (batch_size,), device=device).long()

            # 노이즈 추가
            noise = torch.randn_like(images_4channel).to(device)
            noisy_images = pipe.scheduler.add_noise(images_4channel, noise, timesteps)

            # AMP를 사용한 정밀도 혼합
            with torch.amp.autocast(device_type="cuda"):
                model_output = pipe.unet(noisy_images, timesteps, encoder_hidden_states=text_embeddings)
                loss = torch.nn.functional.mse_loss(model_output.sample, noise)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # Gradient Accumulation
            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            if step % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

        # 에포크 마지막 처리
        if (step + 1) % accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()


fine_tune_model(pipe, dataloader)

def generate_images_with_negative_prompt(prompt, model, negative_prompt, num_images=5):
    images = []
    for i in range(num_images):
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            generated_image = model(
                prompt, 
                negative_prompt=negative_prompt, 
                guidance_scale=8.5, 
                num_inference_steps=50
            ).images[0]
        file_name = f"generated_image_{i+1}.png"
        generated_image.save(file_name)  # 이미지 저장
        images.append(file_name)
        print(f"Image {i+1} saved as {file_name}.")
    return images

# 수정된 프롬프트와 네거티브 프롬프트
prompt = "A serene and peaceful photo of a calm ocean under a clear blue sky, with a majestic sailboat gently drifting on the water, captured in high resolution with soft, natural lighting and vibrant colors."
negative_prompt = "abstract, blurry, low quality, simple texture, plain background"

# 이미지 생성
generated_images = generate_images_with_negative_prompt(prompt, pipe, negative_prompt, num_images=5)

# 생성된 이미지 출력
for image_path in generated_images:
    img = Image.open(image_path)
    img.show()
