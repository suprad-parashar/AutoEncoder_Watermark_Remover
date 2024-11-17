import torch.nn as nn
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

TRAIN_PATH = "/Users/supradparashar/Documents/Suprad/Code/Machine Learning/Datasets/Watermark Images/train"
TEST_PATH = "/Users/supradparashar/Documents/Suprad/Code/Machine Learning/Datasets/Watermark Images/valid"

SHOULD_TRAIN = False
SHOULD_TEST = True

# Hyperparameters
INPUT_SIZE = 350
IN_CHANNELS = 3
OUT_CHANNELS = 16
NUM_CONV_LAYERS = 12
NUM_POOL_LAYERS = 1
DECODER_TYPE = "combined"
BATCH_SIZE = 16
BATCH_NORM = False
MODEL_TYPE = "PaddedPool"

def get_model():
    if MODEL_TYPE == "DiminishingCombined":
        return DiminishingCombinedAutoEncoder(IN_CHANNELS, OUT_CHANNELS, NUM_CONV_LAYERS)
    elif MODEL_TYPE == "DiminishingSplit":
        return DiminishingSplitAutoEncoder(IN_CHANNELS, OUT_CHANNELS, NUM_CONV_LAYERS)
    elif MODEL_TYPE == "PaddedPool":
        return PaddedPoolAutoEncoder(IN_CHANNELS, OUT_CHANNELS, NUM_CONV_LAYERS, NUM_POOL_LAYERS, "center")
    
def get_model_save_name():
    if MODEL_TYPE == "DiminishingCombined":
        return f"DiminishingCombinedAutoEncoder_I{IN_CHANNELS}O{OUT_CHANNELS}C{NUM_CONV_LAYERS}B{BATCH_SIZE}.pth"
    elif MODEL_TYPE == "DiminishingSplit":
        return f"DiminishingSplitAutoEncoder_I{IN_CHANNELS}O{OUT_CHANNELS}C{NUM_CONV_LAYERS}B{BATCH_SIZE}.pth"
    elif MODEL_TYPE == "PaddedPool":
        return f"PaddedPoolAutoEncoder_I{IN_CHANNELS}O{OUT_CHANNELS}C{NUM_CONV_LAYERS}P{NUM_POOL_LAYERS}B{BATCH_SIZE}N{1 if BATCH_NORM else 0}.pth"

def get_latent_space_size_for_diminishing_combined_autoencoder(input_size, num_conv_layers, out_channels):
    # print(input_size)
    for _ in range(num_conv_layers - 1):
        input_size = (input_size - 2)
        input_size = input_size // 2
        # print(input_size)
    return f"{input_size}x{input_size}x{out_channels} = {input_size * input_size * out_channels}"

def get_encoder_decoder_for_padded_pool_autoencoder(in_channels, out_channels, num_conv_layers, num_pool_layers, alignment):
    upper_channels = out_channels * 2 ** (num_conv_layers // 2 - 1)
    # print("UC", upper_channels)
    current_size = INPUT_SIZE
    encoder = nn.Sequential(
        nn.Conv2d(in_channels, upper_channels, 3, stride=1, padding=1),
        nn.ReLU(),
    )
    if alignment == "center":
        conv_batch_size = num_conv_layers // (num_pool_layers + 1)
        current_channels = upper_channels
        for i in range(num_pool_layers):
            for j in range(conv_batch_size // 2):
                encoder.add_module(
                    f"conv_{i}_{j}_1",
                    nn.Conv2d(current_channels, int(current_channels * 0.75), 3, stride=1, padding=1)
                )
                encoder.add_module(
                    f"relu_{i}_{j}_1",
                    nn.ReLU()
                )
                encoder.add_module(
                    f"conv_{i}_{j}_2",
                    nn.Conv2d(int(current_channels * 0.75), current_channels // 2, 3, stride=1, padding=1)
                )
                encoder.add_module(
                    f"relu_{i}_{j}_2",
                    nn.ReLU()
                )
                current_channels //= 2
            encoder.add_module(
                f"maxpool_{i}",
                nn.MaxPool2d(2, stride=2)
            )
            current_size //= 2
            encoder.add_module(
                f"relu_{i}",
                nn.ReLU()
            )
            # # Batch Normalization
            if BATCH_NORM:
                encoder.add_module(
                    f"batchnorm_{i}",
                    nn.BatchNorm2d(current_channels)
                )
                
        for i in range(conv_batch_size // 2 - 1):
            encoder.add_module(
                f"conv_{num_pool_layers}_{i}_1",
                nn.Conv2d(current_channels, int(current_channels * 0.75), 3, stride=1, padding=1)
            )
            encoder.add_module(
                f"relu_{num_pool_layers}_{i}_1",
                nn.ReLU()
            )
            encoder.add_module(
                f"conv_{num_pool_layers}_{i}_2",
                nn.Conv2d(int(current_channels * 0.75), current_channels // 2, 3, stride=1, padding=1)
            )
            encoder.add_module(
                f"relu_{num_pool_layers}_{i}_2",
                nn.ReLU()
            )
            current_channels //= 2
        # print(current_channels)
        decoder = nn.Sequential()
        # print("Size before Decoder Block", current_size)
        for i in range(num_pool_layers):
            for j in range(conv_batch_size // 2):
                decoder.add_module(
                    f"conv_{i}_{j}_1",
                    nn.ConvTranspose2d(current_channels, int(current_channels * 1.5), 3, stride=1, padding=1)
                )
                # print("Conv_{i}_{j}_1", current_size)
                decoder.add_module(
                    f"relu_{i}_{j}_1",
                    nn.ReLU()
                )
                decoder.add_module(
                    f"conv_{i}_{j}_2",
                    nn.ConvTranspose2d(int(current_channels * 1.5), current_channels * 2, 3, stride=1, padding=1)
                )
                decoder.add_module(
                    f"relu_{i}_{j}_2",
                    nn.ReLU()
                )
                current_channels *= 2
            decoder.add_module(
                f"conv_{i}",
                nn.ConvTranspose2d(current_channels, current_channels, 4, stride=2, padding=1)
            )
            current_size = current_size * 2
            decoder.add_module(
                f"relu_{i}",
                nn.ReLU()
            )
        # print("Before Final Decoder Block", current_channels)
        for i in range(conv_batch_size // 2):
            decoder.add_module(
                f"conv_end_{i}_1",
                nn.ConvTranspose2d(current_channels, int(current_channels * 1.5), 3, stride=1, padding=1)
            )
            decoder.add_module(
                f"relu_end_{i}_1",
                nn.ReLU()
            )
            decoder.add_module(
                f"conv_end_{i}_2",
                nn.ConvTranspose2d(int(current_channels * 1.5), current_channels * 2, 3, stride=1, padding=1)
            )
            decoder.add_module(
                f"relu_end_{i}_2",
                nn.ReLU()
            )
            current_channels *= 2
        print("Current Channels", current_size)
        if current_size != INPUT_SIZE:
            print("Hello")
            decoder.add_module(
                f"conv_sizing",
                nn.ConvTranspose2d(current_channels, in_channels, 3, stride=1, padding=3, dilation=3, output_padding=2)
            )
            print("Hello")
        else:
            decoder.add_module(
                f"conv_sizing",
                nn.ConvTranspose2d(current_channels, in_channels, 3, stride=1, padding=1)
            )
        return encoder, decoder

# combined, split
def get_encoder_decoder(in_channels, out_channels, num_conv_layers, decoder_type="combined"):
    channels_upper = out_channels * 2 ** (num_conv_layers - 1)
    encoder = nn.Sequential(
        nn.Conv2d(in_channels, channels_upper, 3, stride=1, padding=0),
        nn.ReLU(),
        nn.MaxPool2d(2, stride=2),
    )
    current_channels = channels_upper
    for i in range(num_conv_layers - 2):
        encoder.add_module(
            f"conv_{i}",
            nn.Conv2d(current_channels, current_channels // 2, 3, stride=1, padding=0)
        )
        current_channels //= 2
        encoder.add_module(
            f"relu_{i}",
            nn.ReLU()
        )
        encoder.add_module(
            f"maxpool_{i}",
            nn.MaxPool2d(2, stride=2)
        )
    encoder.add_module(
        f"conv_{num_conv_layers - 1}",
        nn.Conv2d(current_channels, out_channels, 3, stride=1, padding=0)
    )

    if decoder_type == "split":
        current_channels = out_channels
        decoder = nn.Sequential(
            nn.ConvTranspose2d(current_channels, current_channels * 2, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        current_channels *= 2
        for i in range(num_conv_layers - 2):
            middle_channels = int(current_channels * 1.5)
            decoder.add_module(
                f"conv_{i}_1",
                nn.ConvTranspose2d(current_channels, middle_channels, 4, stride=2, padding=1, output_padding=0)
            )
            decoder.add_module(
                f"relu_{i}_1",
                nn.ReLU()
            )
            decoder.add_module(
                f"conv_{i}_2",
                nn.ConvTranspose2d(middle_channels, current_channels * 2, 3, stride=1, padding=0, output_padding=0)
            )
            decoder.add_module(
                f"relu_{i}_2",
                nn.ReLU()
            )
            current_channels *= 2
        decoder.add_module(
            f"conv_{num_conv_layers - 1}",
            nn.ConvTranspose2d(current_channels, channels_upper, 4, stride=2, padding=1, output_padding=0)
        )
        decoder.add_module(
            f"relu_{num_conv_layers - 1}",
            nn.ReLU()
        )
        decoder.add_module(
            f"conv_{num_conv_layers}",
            nn.ConvTranspose2d(channels_upper, in_channels, 3, stride=1, padding=0, output_padding=0)
        )
    elif decoder_type == "combined":
        current_channels = out_channels
        decoder = nn.Sequential(
            nn.ConvTranspose2d(current_channels, current_channels * 2, 3, stride=1, padding=0),
            nn.ReLU(),
        )
        current_channels *= 2
        for i in range(num_conv_layers - 2):
            middle_channels = int(current_channels * 1.5)
            decoder.add_module(
                f"conv_{i}_1",
                nn.ConvTranspose2d(current_channels, current_channels * 2, 4, stride=2, padding=1, output_padding=2)
            )
            decoder.add_module(
                f"relu_{i}_1",
                nn.ReLU()
            )
            current_channels *= 2
        decoder.add_module(
            f"conv_{num_conv_layers - 1}",
            nn.ConvTranspose2d(current_channels, in_channels, 4, stride=2, padding=1, output_padding=2)
        )
    return encoder, decoder

class WatermarkDataset(Dataset):
    def __init__(self, path):
        self.x_path = path + "/self-watermark"
        self.gt_path = path + "/no-watermark"
        self.files = os.listdir(self.x_path)
        self.transform = transforms.Compose([
            transforms.Resize((350, 350)),
            transforms.ToTensor()
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        watermark_image = self.transform(Image.open(self.x_path + "/" + self.files[idx]))
        original_image = self.transform(Image.open(self.gt_path + "/" + self.files[idx]))
        return watermark_image, original_image
    
class PaddedPoolAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers, num_pool_layers, alignment):
        super(PaddedPoolAutoEncoder, self).__init__()
        self.encoder, self.decoder = get_encoder_decoder_for_padded_pool_autoencoder(in_channels, out_channels, num_conv_layers, num_pool_layers, alignment)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DiminishingCombinedAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(DiminishingCombinedAutoEncoder, self).__init__()
        self.encoder, self.decoder = get_encoder_decoder(in_channels, out_channels, num_conv_layers, "combined")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class DiminishingSplitAutoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv_layers):
        super(DiminishingSplitAutoEncoder, self).__init__()
        self.encoder, self.decoder = get_encoder_decoder(in_channels, out_channels, num_conv_layers, "split")

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    # print(get_latent_space_size(350, NUM_CONV_LAYERS, OUT_CHANNELS))
    # exit()

    # encoder, decoder = get_encoder_decoder_for_padded_pool_autoencoder(IN_CHANNELS, OUT_CHANNELS, NUM_CONV_LAYERS, 2, "center")
    # print(encoder)
    # print(decoder)
    # exit()
    
    model = get_model()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # input("Model Loaded...")

    train_dataset = WatermarkDataset(TRAIN_PATH)
    test_dataset = WatermarkDataset(TEST_PATH)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if SHOULD_TRAIN:
        model.train()
        epochs = 10
        for epoch in range(epochs):
            for watermark_image, original_image in tqdm(train_dataloader):
                watermark_image, original_image = watermark_image.to(device), original_image.to(device)
                optimizer.zero_grad()
                output = model(watermark_image)
                # print("Output Shape:", output.shape)
                # print("Watermark Image Shape:", watermark_image.shape)
                # print("Original Image Shape:", original_image.shape)
                # exit()
                loss = criterion(output, original_image)
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch + 1}/{epochs}, Loss: {loss.item()}")
        torch.save(model.state_dict(), get_model_save_name())

    if SHOULD_TEST:
        model.load_state_dict(torch.load(get_model_save_name(), map_location=device))
        model.eval()
        while True:
            with torch.no_grad():
                i = random.randint(0, len(test_dataset) - 1)
                watermark_image = test_dataset[i][0]
                watermark_image = watermark_image.to(device)
                output = model(watermark_image)
                plt.subplot(1, 2, 1)
                plt.imshow(transforms.ToPILImage()(watermark_image))
                plt.title("Watermark Image")
                plt.subplot(1, 2, 2)
                plt.imshow(transforms.ToPILImage('RGB')(output))
                plt.title("Output Image")
                plt.show()