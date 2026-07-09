import torch
import torch.nn as nn
import torch.nn.functional as F
from model.loss.loss_ssim import ssim
from util.util import YCrCb2RGB

class L_StructureConsistency(nn.Module):
    def __init__(self):
        super(L_StructureConsistency, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, structure_y, function_y, image_fused):
        image_struct = structure_y[:, :1, :, :]
        struct_grad = self.sobelconv(image_struct)
        func_grad = self.sobelconv(function_y)
        fused_grad = self.sobelconv(image_fused)
        joint_grad = torch.max(struct_grad, func_grad)
        return F.l1_loss(joint_grad, fused_grad)
        

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)#.cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)#.cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B))
        Loss_SSIM = weight_A * ssim(image_A, image_fused) + weight_B * ssim(image_B, image_fused)
        return Loss_SSIM

class L_Intensity_MAX(nn.Module):
    def __init__(self):
        super(L_Intensity_MAX, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        
        intensity_joint = torch.max(image_A, image_B)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class L_Intensity_MEAN(nn.Module):
    def __init__(self):
        super(L_Intensity_MEAN, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        image_A = image_A.unsqueeze(0)
        image_B = image_B.unsqueeze(0)
        intensity_joint = torch.mean(torch.cat([image_A, image_B]), dim=0)
        Loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return Loss_intensity

class Fusion_loss(nn.Module):
    def __init__(self,mode='MAX',lambda1=10,lambda2=40,lambda3=40):
        super(Fusion_loss, self).__init__()
        self.mode = str(mode).upper()
        self.lambda1 =lambda1
        self.lambda2 =lambda2
        self.lambda3 =lambda3
        self.L_Struct = L_StructureConsistency()
        if self.mode=='MEAN':
          self.L_Inten = L_Intensity_MEAN()
        if self.mode=='MAX':
          self.L_Inten = L_Intensity_MAX()
        self.L_SSIM = L_SSIM()
   

        # print(1)
    def forward(self, structure_y, function_y, image_fused):
        loss_SSIM = self.lambda1 * (1 - self.L_SSIM(structure_y, function_y, image_fused))
        loss_structure = self.lambda2 * self.L_Struct(structure_y, function_y, image_fused)
        loss_l1 = self.lambda3 * self.L_Inten(structure_y, function_y, image_fused)
        fusion_loss = loss_l1 + loss_structure + loss_SSIM
        return fusion_loss, loss_structure, loss_l1, loss_SSIM
