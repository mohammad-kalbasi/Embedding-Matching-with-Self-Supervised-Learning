
import SimpleITK as sitk
mport torch
import torch.nn as nn
import sys
sys.path.append('/style-augmentation/styleaug')

from ghiasi import Ghiasi
from stylePredictor import StylePredictor
import numpy as np
import sys
from os.path import join, dirname

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StyleAugmentor_Medical_Image(nn.Module):
    def __init__(self):
        super(StyleAugmentor_Medical_Image,self).__init__()

        # create transformer and style predictor networks:
        self.ghiasi = Ghiasi()
        self.stylePredictor = StylePredictor()
        self.ghiasi.to(device)
        self.stylePredictor.to(device)

        # load checkpoints:
        checkpoint_ghiasi = torch.load('/content/style-augmentation/styleaug/checkpoints/checkpoint_transformer.pth',map_location=torch.device(device) )
        checkpoint_stylepredictor = torch.load('/content/style-augmentation/styleaug/checkpoints/checkpoint_stylepredictor.pth',map_location=torch.device(device) )
        checkpoint_embeddings = torch.load('/content/style-augmentation/styleaug/checkpoints/checkpoint_embeddings.pth',map_location=torch.device(device) )

        # load weights for ghiasi and stylePredictor, and mean / covariance for the embedding distribution:
        self.ghiasi.load_state_dict(checkpoint_ghiasi['state_dict_ghiasi'],strict=False)
        self.stylePredictor.load_state_dict(checkpoint_stylepredictor['state_dict_stylepredictor'],strict=False)

        # load mean imagenet embedding:
        self.imagenet_embedding = checkpoint_embeddings['imagenet_embedding_mean'] # mean style embedding for ImageNet
        self.imagenet_embedding = self.imagenet_embedding.to(device)

        # get mean and covariance of PBN style embeddings:
        self.mean = checkpoint_embeddings['pbn_embedding_mean']
        self.mean = self.mean.to(device) # 1 x 100
        self.cov = checkpoint_embeddings['pbn_embedding_covariance']

        # compute SVD of covariance matrix:
        u, s, vh = np.linalg.svd(self.cov.numpy())

        self.A = np.matmul(u,np.diag(s**0.5))
        self.A = torch.tensor(self.A).float().to(device) # 100 x 100
        # self.cov = cov(Ax), x ~ N(0,1)

    def sample_embedding(self,n):
        # n: number of embeddings to sample
        # returns n x 100 embedding tensor
        embedding = torch.randn(n,100).to(device) # n x 100
        embedding = torch.mm(embedding,self.A.transpose(1,0)) + self.mean # n x 100
        return embedding

    def forward(self,x,alpha=0.5,downsamples=0,embedding=None,useStylePredictor=True,refrence_idx = None):
        # augments a batch of images with style randomization
        # x: B x C x H x W image tensor
        # alpha: float in [0,1], controls interpolation between random style and original style
        # downsamples: int, number of times to downsample by factor of 2 before applying style transfer
        # embedding: B x 100 tensor, or None. Use this embedding if provided.
        # useStylePredictor: bool. If True, we use the inception based style predictor to compute the original style embedding for the input image, and use that for interpolation. If False, we use the mean ImageNet embedding instead, which is slightly faster.
        if refrence_idx == None:
                  refrence_idx = x.shape[-1]//2
        refrence_x = x[:,:,:,:,refrence_idx].squeeze(-1)

        # style embedding for when alpha=0:
        base = self.stylePredictor(refrence_x) if useStylePredictor else self.imagenet_embedding



        if downsamples:
            assert(refrence_x.size(2) % 2**downsamples == 0)
            assert(refrence_x.size(3) % 2**downsamples == 0)
            for i in range(downsamples):
                x = nn.functional.avg_pool2d(x,2)

        if embedding is None:
            # sample a random embedding
            embedding = self.sample_embedding(refrence_x.size(0))
        # interpolate style embeddings:
        embedding = alpha*embedding + (1-alpha)*base
        shape_sample = self.ghiasi(refrence_x,embedding)
        n0,n1,n2,n3 = shape_sample.shape
        n4 = x.shape[-1]
        restyled = torch.zeros((n0,1,n2,n3,n4)).to(device)
        print(restyled.shape)
        print((rgb_to_gray_tensor(self.ghiasi(x[:,:,:,:,0].squeeze(-1),embedding))).shape)
        for i in range(x.size(-1)):
          restyled[:,:,:,:,i] = rgb_to_gray_tensor(self.ghiasi(x[:,:,:,:,i].squeeze(-1),embedding))

        if downsamples:
            restyled = nn.functional.upsample(restyled,scale_factor=2**downsamples,mode='bilinear')

        return restyled.detach()
class Elastic_transform(nn.Module):
  def __init__(self,size,control_points,max_displacement,num_dimensions = 3,num_locked_borders =2,SPLINE_ORDER = 3):
    super().__init__()
    self.size = size
    self.control_points = control_points
    self.max_displacement = max_displacement
    self.num_dimensions = num_dimensions
    self.num_locked_borders = num_locked_borders
    self.SPLINE_ORDER = SPLINE_ORDER
    self.num_control_points = np.array(control_points, np.uint32)
    self.mesh_size = self.num_control_points - self.SPLINE_ORDER
    img_template_size = np.zeros(size)
    self.img_template = sitk.GetImageFromArray(img_template_size)
    self.spt_trans = SpatialTransformer(size)
  def forward(self,img):
    grid_shape = self.num_control_points
    coarse_field = torch.rand(*grid_shape, self.num_dimensions)  # [0, 1)
    coarse_field -= 0.5  # [-0.5, 0.5)
    coarse_field *= 2  # [-1, 1]
    for dimension in range(3):
        # [-max_displacement, max_displacement)
        coarse_field[..., dimension] *= self.max_displacement[dimension]
    # Set displacement to 0 at the borders
    for i in range(self.num_locked_borders):
        coarse_field[i, :] = 0
        coarse_field[-1 - i, :] = 0
        coarse_field[:, i] = 0
        coarse_field[:, -1 - i] = 0
    num_control_points = coarse_field.shape[:-1]
    mesh_shape = [n - self.SPLINE_ORDER for n in num_control_points]
    bspline_transform = sitk.BSplineTransformInitializer(self.img_template, mesh_shape)
    coarse_field = coarse_field.numpy()
    parameters = coarse_field.flatten(order='F').tolist()
    bspline_transform.SetParameters(parameters)
    flow_real = sitk.TransformToDisplacementField(bspline_transform
                                    ,sitk.sitkVectorFloat64,
                                    self.img_template.GetSize(),
                              self.img_template.GetOrigin(),
                              self.img_template.GetSpacing(),
                              self.img_template.GetDirection())
    flow_real = torch.tensor(sitk.GetArrayFromImage(flow_real).transpose(3,0,1,2)).float()

    out_spat = self.spt_trans(img.unsqueeze(0),flow_real.unsqueeze(0))
    return out_spat.squeeze(0),flow_real.squeeze()
