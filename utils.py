from torchvision import transforms
from skimage import color
import numpy as np
from PIL import Image
import math
import torch
import torch.nn.functional as F

def convertLAB2RGB( lab ):
   lab[:, :, 0:1] = lab[:, :, 0:1] * 100   # [0, 1] -> [0, 100]
   lab[:, :, 1:3] = np.clip(lab[:, :, 1:3] * 255 - 128, -100, 100)  # [0, 1] -> [-128, 128]
   rgb = color.lab2rgb( lab.astype(np.float64) )
   return rgb

def convertRGB2LABTensor( rgb ):
   lab = color.rgb2lab( np.asarray( rgb ) ) # RGB -> LAB L[0, 100] a[-127, 128] b[-128, 127]
   ab = np.clip(lab[:, :, 1:3] + 128, 0, 255) # AB --> [0, 255]
   ab = transforms.ToTensor()( ab ) / 255.
   L = lab[:, :, 0] * 2.55 # L --> [0, 255]
   L = Image.fromarray( np.uint8( L ) )
   L = transforms.ToTensor()( L ) # tensor [C, H, W]
   return L, ab.float()

def addMergin(img, target_w, target_h, background_color=(0,0,0)):
   width, height = img.size
   if width==target_w and height==target_h:
      return img
   scale = max(target_w,target_h)/max(width, height)
   width = int(width*scale/16.)*16
   height = int(height*scale/16.)*16
   img = transforms.Resize( (height,width), interpolation=Image.BICUBIC )( img )

   xp = (target_w-width)//2
   yp = (target_h-height)//2
   result = Image.new(img.mode, (target_w, target_h), background_color)
   result.paste(img, (xp, yp))
   return result


def psnr(ref, img):
   '''
   Peak signal-to-noise ratio (PSNR).
   '''
   if type(ref) is torch.Tensor:
      mse = torch.mean((ref - img) ** 2)
      if mse == 0:
         return 100
      PIXEL_MAX = 1.
      return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

   mse = np.mean((ref - img) ** 2)
   if mse == 0:
      return 100
   PIXEL_MAX = 1.
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def ssim(img1, img2, val_range=255, window_size=11, window=None, size_average=True, full=False):
   def gaussian(window_size, sigma):
      gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
      return gauss / gauss.sum()

   def create_window(window_size, channel=1):
      # Generate an 1D tensor containing values sampled from a gaussian distribution
      _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
      # Converting to 2D
      _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)

      window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

      return window

   L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

   pad = window_size // 2

   try:
      _, channels, height, width = img1.size()
   except:
      channels, height, width = img1.size()

   # if window is not provided, init one
   if window is None:
      real_size = min(window_size, height, width)  # window should be atleast 11x11
      window = create_window(real_size, channel=channels).to(img1.device)

   # calculating the mu parameter (locally) for both images using a gaussian filter
   # calculates the luminosity params
   mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
   mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

   mu1_sq = mu1 ** 2
   mu2_sq = mu2 ** 2
   mu12 = mu1 * mu2

   # now we calculate the sigma square parameter
   # Sigma deals with the contrast component
   sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
   sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
   sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

   # Some constants for stability
   C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
   C2 = (0.03) ** 2

   contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
   contrast_metric = torch.mean(contrast_metric)

   numerator1 = 2 * mu12 + C1
   numerator2 = 2 * sigma12 + C2
   denominator1 = mu1_sq + mu2_sq + C1
   denominator2 = sigma1_sq + sigma2_sq + C2

   ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

   if size_average:
      ret = ssim_score.mean()
   else:
      ret = ssim_score.mean(1).mean(1).mean(1)

   if full:
      return ret, contrast_metric

   return ret
