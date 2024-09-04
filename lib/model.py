import torch.nn as nn
import torch
from torchsummary import summary
from config.config import Config

class PatchGanBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=4,stride=2,padding=1,initial_layer=False):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels,out_channels,kernel_size,stride,padding,bias=initial_layer,
                padding_mode='reflect'
            ))
        if not initial_layer:
            self.conv_block.append(
                nn.InstanceNorm2d(out_channels)
                )
        self.conv_block.append(
            nn.LeakyReLU(0.2)
        )        
    
    def forward(self,x):
        return self.conv_block(x)


# Patch Gan discriminator : from (3,256,256) to (1,30,30) : patch of 30
class Discriminator(nn.Module):
    def __init__(self,features=[3,64,128,256,512]) -> None:
        super().__init__()
        self.conv_blocks = self._create_conv_blocks(features)

    def _create_conv_blocks(self,features)->nn.Module:
        conv_blocks = nn.Sequential()
        in_channels = features[0]
        for i,out_channles in enumerate(features[1:]):
            stride = 1 if out_channles==features[-1] else 2
            conv_blocks.append(
                PatchGanBlock(in_channels,out_channles,stride=stride,initial_layer=i==0) 
            )
            in_channels=out_channles

        conv_blocks.append(
            nn.Conv2d(features[-1],1,4,1,1,padding_mode='reflect')
        )
        return conv_blocks
    
    def forward(self,x):
        return self.conv_blocks(x)


#### The Generator

class GeneratorConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,
                 down:bool,use_activ:bool=True,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode='reflect',bias=False,**kwargs) if down
            else nn.ConvTranspose2d(in_channels,out_channels,bias=False,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_activ else nn.Identity()
        )

    def forward(self,x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_channles=256):
        super().__init__()
        self.block = nn.Sequential(
            GeneratorConvBlock(
                use_activ=True,in_channels=in_channles,out_channels=in_channles,\
                down=True,kernel_size=3,stride=1,padding=1
            ),
            GeneratorConvBlock(
                use_activ=False,in_channels=in_channles,out_channels=in_channles,\
                down=True,kernel_size=3,stride=1,padding=1
            ),
        )

    def forward(self,x):
        return x+self.block(x)
    
class Generator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.initial_conv_block = nn.Sequential(
            GeneratorConvBlock(down=True,in_channels=3,\
                               out_channels=64,kernel_size=7,stride=1,padding=3),
            GeneratorConvBlock(down=True,in_channels=64,\
                               out_channels=128,kernel_size=3,stride=2,padding=1),
            GeneratorConvBlock(down=True,in_channels=128,\
                               out_channels=256,kernel_size=3,stride=2,padding=1),            
        )

        self.residual_blocks = self._make_residual_blocks()

        self.up_blocks = self._make_up_blocks()

        self.last_layer = nn.Conv2d(
            3,3,3,1,1,padding_mode='reflect'
        )

    def forward(self,x):
        x=self.initial_conv_block(x)
        x=self.residual_blocks(x)
        return self.up_blocks(x)

    def _make_residual_blocks(self,nb_res_blocks=3)->nn.Sequential:
        res_blocks=nn.Sequential()
        for _ in range(nb_res_blocks):
            res_blocks.append(ResidualBlock())
        return res_blocks
    
    def _make_up_blocks(self)->nn.Sequential:
        up_blocks = nn.Sequential()
        features=[256,128,64,3]

        in_channels = features[0]    
        for out_channels in features[1:-1:1]:
            up_blocks.append(
                GeneratorConvBlock(
                    in_channels=in_channels,out_channels=out_channels,\
                        down=False,use_activ=True,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        output_padding=1
                )
            )
            in_channels=out_channels
        up_blocks.append(
            nn.Conv2d(
                64,3,7,1,3,padding_mode='reflect'
            )
        )
        up_blocks.append(
            nn.Tanh()
        )
        return up_blocks
    
def get_model_optimizer():
    gen = Generator().to(Config.device)
    disc = Discriminator().to(Config.device)

    gen_optim = torch.optim.Adam(params=gen.parameters(),lr=Config.learning_rate)
    disc_optim = torch.optim.Adam(params=disc.parameters(),lr=Config.learning_rate)

    return gen,disc,gen_optim,disc_optim

def get_losses():
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    return l1_loss_fn,mse_loss_fn

if __name__=="__main__":
    disc = Discriminator().to('cuda')
    gen = Generator().to('cuda')
    print(summary(
        gen,
        (3,256,256)
    ))
