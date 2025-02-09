import torch
from model import VTransformer, settings, TumorDataset, get_dataloader
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn.functional as F

d = {
	'N':0,
	'G':1,
	'M':2,
	'P':3
}

d2 = {
	0:'N',
	1:'G',
	2:'M',
	3:'P'
}

t = transforms.Compose([
    transforms.Resize((settings['img_size'],settings['img_size'])),
    transforms.ToTensor()
])


def prep_img(path):
	img = Image.open(path).convert('L')
	img = t(img)
	label = d.get(path.split('\\')[-1][0],None)
	img = img.unsqueeze(0) #add batch dim
	return img,label


path = os.path.join('Data','Tumor','pituitary_tumor','P_66_HF_.jpg')
print(path)
x,y = prep_img(path)
model = VTransformer(**settings)

model.load_state_dict(torch.load('w8L.pth', weights_only=True))
model.eval()

rs = model(x)
rs = F.softmax(rs, dim=1)

rs = [f'{d2[i]}={rs[0][i].item():.2f}' for i in range(len(rs[0]))]
print(rs, f'\nReal label is {d2[y]}')