import torch
from model import VTransformer, settings, TumorDataset, get_dataloader
from PIL import Image
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
from random import choice
import numpy as np
import matplotlib.pyplot as plt
import cv2

def get_rollout(model):


	rollout = torch.eye(model.attention[0].att.att_w.size(-1)).to(model.attention[0].att.att_w.device)

	for block in model.attention:
		attention = block.att.att_w

		#avg among heads => A
		attention_heads_fused = attention.mean(dim=1)  
		#adding identity (wtf?) => A + I
		attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
		#aggregation
		attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True) #norm A
		rollout = torch.matmul(rollout, attention_heads_fused)


	return rollout
 


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



path = os.path.join('Data2','Testing')
cats = os.listdir(path)
cats.remove("notumor")
cat = choice(cats)
path = os.path.join(path,cat)
fs = os.listdir(path)
path = os.path.join(path,choice(fs))

print(path)
x,y = prep_img(path)
model = VTransformer(**settings)

model.load_state_dict(torch.load('w8L.pth', weights_only=True))
model.eval()

rs = model(x)
rs = F.softmax(rs, dim=1)

rs = [f'{d2[i]}={rs[0][i].item():.2f}' for i in range(len(rs[0]))]
print(rs, f'\nReal label is {d2.get(y,None)}')


#attention rollout section
num_of_patches = settings['img_size'] // settings['patch_size']

rollout = get_rollout(model)
cls_att = rollout[0,1:,0]
#contains heatmap in numpy arr
cls_att = 1 - cls_att.reshape(num_of_patches, num_of_patches)


#reopen img
original_img = Image.open(path).convert('RGB')
original_img = original_img.resize((224, 224))
original_img = np.array(original_img)
print(type(cls_att))
heatmap = cv2.resize(cls_att.detach().numpy(), (224, 224))
#heatmap[heatmap<0.2] = 0 #in case we need filtering activity
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
overlay= cv2.hconcat([overlay,original_img])
plt.figure(figsize=(8, 8))
plt.imshow(overlay)
plt.axis("off")
plt.title(f"Pred: {rs}\nReal label is {d2.get(y,None)}")
plt.show()