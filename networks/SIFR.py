import torch
import torchvision
from torch import nn
from functools import partial

class ResidualBlock(nn.Module):
    """
    Implements Residual Block
    
    Parameters 
    ----------
    n_channels : int 
                Number of channels(Input and Output)
    pad : 
                Type of padding to choose
    norm_layer : 
                Type of normalisation to choose
    activation : 
                Activation function to choose
    use_dropout : Bool
                Use Dropout or not

    Attributes 
    ----------
    block : List 
                List of all Modules to use 
    """

    def __init__(self , n_channels , pad, norm_layer , activation , use_dropout = False):
        """
        Initializes the block 
        """
        super().__init__()
        # Input : (batch_size , channels , height , width) :: Output : (batch_size , channels, height , width)
        block = [pad(1),
                nn.Conv2d(n_channels, n_channels , kernel_size = 3 , padding = 0 , stride = 1),
                norm_layer(n_channels),
                activation]
        # Use Dropout 
        if use_dropout:
            block += [nn.Dropout(0.5)]
        # Input : (batch_size , channels , height , width) :: Output : (batch_size , channels,  height , width)
        block += [pad(1),
                nn.Conv2d(n_channels , n_channels , kernel_size = 3 , padding = 0 , stride = 1),
                norm_layer(n_channels)]
        # Deferencing 
        self.block = nn.Sequential(*block)

    def forward(self , x):
        """
        Implements the forward method 
        
        Parameters
        ----------
        x : torch.tensor
                Input feature volume , Shape = (batch_size , channel , height , widht)
        Returns 
        -------
        torch.Tensor 
                Output feature volume after residual block  , Shape = (batch_size , channel , height , widht)
        """
        return x + self.block(x)

#------------------------------
# Sets the padding method for the input
def get_pad_layer(type):
    # Chooses reflection , places mirror around boundary and reflects the value
    if type == "reflection":
        layer = nn.ReflectionPad2d
    # Replicates the padded area with nearest boundary value
    elif type == "replication":
        layer = nn.ReplicationPad2d
    # Padding of Image with constat 0 
    elif type == "zero":
        layer = nn.ZeroPad2d
    else:
        raise NotImplementedError("Padding type {} is not valid . Please choose among ['reflection' ,'replication' ,'zero']".format(type))
    
    return layer
    
#----------------------------------
# Sets the norm layer 
def get_norm_layer(type):
    if type == "BatchNorm2d":
        layer = partial(nn.BatchNorm2d , affine = True) 
    elif type == "InstanceNorm2d":
        layer = partial(nn.InstanceNorm2d ,affine = False)
    else : 
        raise NotImplementedError("Norm type {} is not valid. Please choose ['BatchNorm2d' , 'InstanceNorm2d']".format(type))
    
    return layer


class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.model =torchvision.models.vgg16(pretrained=True).features[:30].eval()
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self , x):
        return self.model(x)


class EncoderDecoder(nn.Module):
    """
    Implements Encoder Decoder 

    Parameters
    ----------
    opt : 
        User specified arguments
    input_ch : int 
        Number of input channels 
    output_ch : int 
        Number of output channels 
    Attributes 
    ----------
    n_edf : int 
        Number of channels after first conv layer
    norm : 
        Type of norm to use
    activation : 
        Type of activation to use
    model : 
        List containing all architecture modules 

    """    
    def __init__(
        self,
        input_ch,
        output_ch,
        n_edf=16,
        init_mean=0.0,
        init_std=0.02,
        ):
        """
        Initializes the module architecture
        """
        super(EncoderDecoder, self).__init__()
        norm = nn.InstanceNorm2d
        activation = nn.ReLU(inplace=True)

        model = []
        model += [nn.Conv2d(input_ch , n_edf , kernel_size=7 , stride = 1 , padding = 3) , norm(n_edf) , activation]
        model += [nn.Conv2d(n_edf , n_edf * 4 , kernel_size= 3 , padding = 1 , stride= 2) , norm(n_edf * 4) , activation]
        model += [nn.Conv2d(n_edf * 4 , n_edf * 8 , kernel_size= 3 , padding = 1 , stride= 2) , norm(n_edf * 8) , activation]
        model += [nn.Conv2d(n_edf * 8 , n_edf * 16 , kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf * 16) , activation]
        model += [nn.Upsample(scale_factor=2 , mode="nearest"),
                 nn.Conv2d(n_edf * 16 , n_edf * 8 , kernel_size=3 , stride = 1 , padding =1),
                 norm(n_edf * 8) ,
                 activation]
        model += [nn.Upsample(scale_factor=2 , mode="nearest"),
                 nn.Conv2d(n_edf * 8 , n_edf * 4 , kernel_size=3 ,
                  stride = 1 , padding =1) , norm(n_edf * 4) ,
                   activation]
        model += [nn.Conv2d(n_edf * 4 , n_edf * 2, kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf * 2) , activation]
        model += [nn.Conv2d(n_edf * 2 , n_edf, kernel_size= 3 ,padding = 1 , stride = 1), norm(n_edf) , activation]
        model += [nn.Conv2d(n_edf , output_ch, kernel_size= 7 ,padding = 3 , stride = 1)]

        self.model = nn.Sequential(*model)        
        self.activation = torch.nn.Sigmoid()

    def forward(self , x):
        return self.activation(self.model(x))

class Generator(nn.Module):
    """
    Implements Generator

    Parameters
    ----------
    opt : 
        User specified arguments
    input_ch : int 
        Number of channels in the input
    output_ch : int 
        Number of channels in Output
    
    Attributes 
    ----------
    model : 
            List of all network modules
    n_gf : int 
            Number of output channel of first convolution of gen
    pad : 
            Type of padding to use
    norm : 
            Type of norm to use
    """
    def __init__(
        self,
        input_ch,
        output_ch,
        n_gf=64,
        norm_type="BatchNorm2d",
        pad_type="reflection",
        n_downsample=2,
        n_residual=9,
        init_mean=0.0,
        init_std=0.02,
        ):
        """
        Initiates the generator
        """
        super(Generator, self).__init__()

        activation = nn.ReLU()
        n_gf = n_gf
        norm = get_norm_layer(norm_type)
        pad = get_pad_layer(pad_type)

        model = []
        # Same shape as input 
        model += [pad(3) , nn.Conv2d(input_ch , n_gf , kernel_size = 7 , padding = 0) , norm(n_gf) , activation]
        # Downsampling the input
        for num in range(n_downsample):
            model += [nn.Conv2d(n_gf , 2 * n_gf , kernel_size = 3 , padding =1 ,  stride = 2) , norm(n_gf * 2) , activation]
            n_gf = n_gf * 2
        # Same shape as the input
        for num in range(n_residual):
            model += [ResidualBlock(n_gf , pad , norm , activation)]
        # Upsampling the input
        for num in range(n_downsample):
            model += [nn.Upsample(scale_factor = 2 , mode="nearest") ,nn.Conv2d(n_gf ,n_gf // 2 , kernel_size = 3 , padding = 1 , stride = 1) ,norm(n_gf // 2) , activation ]
            n_gf = n_gf //2
        # Same shape as the input 
        model += [pad(3) , nn.Conv2d(n_gf , output_ch , kernel_size = 7 , stride = 1 , padding = 0)]
        self.model = nn.Sequential(*model)

    def forward(self , x):
        return self.model(x)

class PatchDiscriminator(nn.Module):
    """
    Implements PatchGAN discriminator 

    Parameters
    ---------
    input_ch : int
            Input channels to the discriminator 
    opt :   
            User specififed arguments
    
    Attributes 
    ---------
    n_df : int 
            Discriminator features after first layer
    norm : 
            Type of norm to use
    activation : 
            Type of activation to use
    """
    def __init__(
        self,
        input_ch,
        n_df=64,
        init_mean=0.0,
        init_std=0.02,
        ):
        super(PatchDiscriminator, self).__init__()
        # Required attributes
        activation = nn.LeakyReLU(0.2 , inplace=True)
        norm = nn.InstanceNorm2d
        # Block for model
        block= []
        block+= [nn.Conv2d(input_ch , n_df , kernel_size = 4 , stride = 2 , padding =1) , activation]
        block+= [nn.Conv2d(n_df , n_df *2 , kernel_size = 4, stride =2 , padding = 1) , norm(n_df * 2) , activation]
        block+= [nn.Conv2d(n_df * 2, n_df * 4 , kernel_size = 4  , padding = 1 , stride = 2) ,norm(n_df * 4) , activation]
        block+= [nn.Conv2d(n_df*4 , n_df * 8 , kernel_size = 4 , stride = 1 , padding = 1) , norm(n_df * 8) , activation]
        block+= [nn.Conv2d(n_df*8 , 1 ,kernel_size=4 , stride = 1 , padding = 1)]
        # Dereferencing block
        self.block = nn.Sequential(*block)
        # Defining activation
        self.activation = torch.nn.Sigmoid()

    def forward(self , x):
        return self.activation(self.block(x))

class SIFRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SIFRNet, self).__init__()
        self.FR = Generator(4 , 3)
        self.FG = Generator(4 , 3)
        self.LSD = EncoderDecoder(3 , 1)
        self.FD = EncoderDecoder(3 , 1)
        self.D_flarefree = PatchDiscriminator(3)
        self.D_withflare = PatchDiscriminator(3)

    def forward(self, x):
        flare_mask = self.FD(x)
        fr_input = torch.cat((flare_mask, x), dim=1)
        return self.FR(fr_input)
    
    def forward_flarefree(self, x):
        light_source_mask = self.LSD(x)
        generate_flare_image = self.FG(torch.cat((light_source_mask, x), dim=1))

        flare_mask = self.FD(generate_flare_image)
        rec_image = self.FR(torch.cat((flare_mask, generate_flare_image), dim=1))

        light_source_mask_fake = self.LSD(rec_image)
        d_rec = self.D_flarefree(rec_image)
        flare_mask_fake = self.FD(rec_image)

        return rec_image, flare_mask, flare_mask_fake, light_source_mask, light_source_mask_fake, d_rec

    def forward_withflare(self, x):
        flare_mask = self.FD(x)
        generate_flare_free_image = self.FR(torch.cat((flare_mask, x), dim=1))
        light_source_mask = self.LSD(x)
        rec_image = self.FG(torch.cat((light_source_mask, generate_flare_free_image), dim=1))

        light_source_mask_fake = self.LSD(generate_flare_free_image)
        d_rec = self.D_withflare(rec_image)
        flare_mask_fake = self.FD(rec_image)

        return generate_flare_free_image, rec_image, flare_mask, flare_mask_fake, light_source_mask, light_source_mask_fake, d_rec



def _test():
    torch.set_grad_enabled(False)
    from torchinfo import summary

    sifr = SIFRNet(3, 3)
    print(sifr)
    summary(sifr, input_size=(1, 3, 512, 512), device="cpu")
    x = torch.randn(1, 3, 256, 256)
    x = (x - x.min()) / (x.max() - x.min())
    y = sifr(x)
    print(x.size(), x.min(), x.max(), x.mean())
    print(y.size(), y.min(), y.max(), y.mean())
    rec_image, flare_mask, flare_mask_fake, light_source_mask, light_source_mask_fake, d_rec = sifr.forward_withflare(x)
    print(flare_mask.size())


if __name__ == "__main__":
    _test()