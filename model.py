import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000,j/emb_size))) if j%2==0  else np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return torch.tensor(embeddings)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768, img_size=224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size,stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
            ) 
        self.cls_token = nn.Parameter(torch.rand(1,1,emb_size))
        self.pos_embed = nn.Parameter(PositionEmbedding((img_size//patch_size)**2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        
        cls_token = repeat(self.cls_token, ' () s e -> b s e', b=b)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        return x


class MultiHead(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.key = nn.Linear(emb_size,emb_size)
        self.value = nn.Linear(emb_size,emb_size)
        self.query = nn.Linear(emb_size,emb_size)
        self.att_dr = nn.Dropout(0.1)


    def forward(self,x):
        k = rearrange(self.key(x), 'b n (h e) -> b h n e', h =self.num_head)
        q = rearrange(self.query(x), 'b n (h e) -> b h n e', h =self.num_head)
        v = rearrange(self.value(x), 'b n (h e) -> b h n e', h =self.num_head)

        wei = q@k.transpose(3,2)/self.num_head**0.5
        wei = F.softmax(wei, dim=2)
        wei = self.att_dr(wei)

        out = wei@v 
        out = rearrange(out, 'b h n e -> b n (h e)')
        return out


class FeedForward(nn.Module):
    def __init__(self,emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.ReLU(),
            nn.Linear(4*emb_size, emb_size)
            )

    def forward(self,x):
        return self.ff(x)


class Block(nn.Module):
    def __init__(self, emb_size,num_head):
        super().__init__()
        self.att = MultiHead(emb_size, num_head)
        self.ll = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)
        self.ff =FeedForward(emb_size)

    def forward(self,x):
        #print(x)
        xn = self.ll(x)
        #print(xn)
        ax = self.att(xn)
        #print(ax)
        x = x + self.dropout(ax)
        #print(x)
        xn = self.ll(x)
        #print(xn)
        x = x + self.dropout(self.ff(xn))
        #print(x)
        return x


class VTransformer(nn.Module):
    def __init__(self,
     in_channels,
     num_layers,
     img_size,
     emb_size,
     patch_size,
     num_head,
     num_class):

        super().__init__()
        self.attention = nn.Sequential(
            *[Block(emb_size, num_head) for _ in range(num_layers)]
            )

        #in_channels=3, patch_size=16, emb_size=768, img_size=224):
        self.patchemb = PatchEmbedding(in_channels=in_channels, emb_size=emb_size, patch_size=patch_size, img_size=img_size)
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self,x):
        embeddings = self.patchemb(x)
        x = self.attention(embeddings)
        x = self.ff(x[:,0,:])
        return x

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"USED DEVICE: {device}")
settings = dict(
    in_channels = 1,
    num_layers = 8, 
    emb_size = 384, 
    num_head = 12, 
    num_class = 4, 
    patch_size = 16, 
    img_size = 224
)








class TumorDataset(Dataset):
    def __init__(self,paths, labels, transform=None):
        self.transform = transform
        self.paths = paths
        self.labels = labels


    def __len__(self):
        return len(self.paths)

    def __getitem__(self,i):
        img = self.paths[i]
        label = self.labels[i]
        img = Image.open(img).convert('L')
        if self.transform:
            img = self.transform(img)
        img = img.float()
        return img,label

def get_dataloader(paths,labels,img_size,batch_size=32, shuffle=True):
    t = transforms.Compose([
        transforms.Resize((settings['img_size'],settings['img_size'])),
        transforms.ToTensor()
    ])

    dataset = TumorDataset(paths,labels,t)
    return DataLoader(dataset,batch_size=batch_size, shuffle=True)


def load_and_split_data(subsets, img_size, test_size=0.2, batch_size=32):
    image_paths = []
    labels = []

    for folder, label in subsets.items():
        for filename in os.listdir(folder):
            image_paths.append(os.path.join(folder, filename))
            labels.append(label)

    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels, test_size=test_size, random_state=42)
    train_loader = get_dataloader(X_train, y_train, img_size, batch_size)
    test_loader = get_dataloader(X_test, y_test, img_size, batch_size, shuffle=False)
    return train_loader, test_loader    








if __name__ == "__main__":


    subsets = {
    os.path.join('Data','Normal') : 0,
    os.path.join('Data','Tumor','glioma_tumor') : 1,
    os.path.join('Data','Tumor','meningioma_tumor'): 2,
    os.path.join('Data','Tumor','pituitary_tumor'): 3
    }


    print("Brain Tumor Dataset\n")

    train_loader, test_loader = load_and_split_data(subsets, settings['img_size'], test_size=0.2, batch_size=32)
    model = VTransformer(**settings)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 30
    #torch.set_num_threads(2)


    L = settings['num_layers']
    H = settings['num_head']
    E = settings['emb_size']
    ml_fname = f'{L}L-{E}E-{H}H.pth'

    print(ml_fname)


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in  tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            out = model(inputs)
    
            loss = criterion(out, labels)
    
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            _, pred = torch.max(out,1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total * 100 
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
        model.eval()  
        with torch.no_grad(): 
            test_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in tqdm(test_loader, desc="Testing"):
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                loss = criterion(out, labels)
                test_loss += loss.item()
    
                _, predicted = torch.max(out, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
    
            test_loss = test_loss / len(test_loader)
            test_accuracy = correct / total * 100
            print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
        fin_acc = epoch_accuracy
    
    
    
    torch.save(model.state_dict(), ml_fname)