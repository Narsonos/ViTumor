import torch
from model import VTransformer, settings, TumorDataset, get_dataloader
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from random import choice

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

d3 = {
	'notumor':'N',
	'glioma':'G',
	'meningioma':'M',
	'pituitary':'P'
}

t = transforms.Compose([
    transforms.Resize((settings['img_size'],settings['img_size'])),
    transforms.ToTensor()
])


def prep_img(path, label=None):
	img = Image.open(path).convert('L')
	img = t(img)
	if not label:
		label = d.get(path.split('\\')[-1][0],None)
		if not label:
			label = d.get(d3.get(path.split('\\')[-2]),None)
	img = img.unsqueeze(0) #add batch dim
	return img,label


#path = os.path.join('Data','Tumor','pituitary_tumor','P_66_HF_.jpg')
path = os.path.join('Data2','Testing')

cat = choice(os.listdir(path))
path = os.path.join(path,cat)
fs = os.listdir(path)
path = os.path.join(path,choice(fs))


#path = 'P3.jfif'
print(path)
x,y = prep_img(path)
model = VTransformer(**settings)

model.load_state_dict(torch.load('w8L.pth', weights_only=True))
model.eval()

rs = model(x)
rs = F.softmax(rs, dim=1)

rs = [f'{d2[i]}={rs[0][i].item():.2f}' for i in range(len(rs[0]))]
print(rs, f'\nReal label is {d2.get(y,None)}')