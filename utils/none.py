import torch
import torch.nn.functional as F
pred=torch.randn(3,4,5)
print(pred)
pred =torch.mean(pred, dim=1, keepdim=True)
print(pred)
masks_context =  (pred > 0).float().unsqueeze(1)
print(masks_context)
masks_context = F.interpolate(masks_context, size=(10,10), mode='nearest')
print(masks_context)