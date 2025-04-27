import torch
import torch.nn as nn

import torch
import torch.nn as nn

#sinusoidal position embedding(section 3.3)
#basically a positional encoding to give each timestep a distinct representation.
def get_time_embedding(time_steps, t_emb_dim):
    factor=10000**((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device)/(t_emb_dim//2)))
    t_emb=time_steps[:, None].repeat(1, t_emb_dim//2)/factor
    t_emb=torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb

#downblock
class DownSampleBlock(nn.Module):
  def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
    super().__init__()
    self.down_sample=down_sample
    self.resnet_conv_first=nn.Sequential(
        nn.GroupNorm(8, in_channels),
        nn.SiLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    )

    self.t_emb_layers=nn.Sequential(
        nn.SiLU(),
        nn.Linear(t_emb_dim, out_channels)
    )

    self.resnet_conv_second=nn.Sequential(
        nn.GroupNorm(8, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    )

    self.attention_norm=nn.GroupNorm(8, out_channels)
    self.attention=nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
    self.residual_input_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
    self.down_sample_conv=nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                    stride=2, padding=1) if self.down_sample else nn.Identity()

  def forward(self, x, t_emb):
    out = x
    #RESNET BLOCK
    resnet_input = out
    out=self.resnet_conv_first(out)

    #time embedding
    out=out+self.t_emb_layers(t_emb)[:, :, None, None]
    out=self.resnet_conv_second(out)

    #residual input
    out=out+self.residual_input_conv(resnet_input)

    #ATTENTION BLOCK
    batch_size, channels, height, width=out.shape
    in_attn=out.reshape(batch_size, channels, height*width)
    in_attn=self.attention_norm(in_attn)
    in_attn=in_attn.transpose(1,2)
    out_attn, _ = self.attention(in_attn, in_attn, in_attn)
    out_attn=out_attn.transpose(1,2).reshape(batch_size, channels, height, width)
    out=out+out_attn

    #downsample
    out=self.down_sample_conv(out)
    return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        self.resnet_conv_first=nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.t_emb_layers=nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        self.resnet_conv_second=nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(8, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.attention_norm=nn.GroupNorm(8, out_channels)
        self.attention=nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv=nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])
    

    def forward(self, x, t_emb):
        out = x
        #first resnet block
        resnet_input=out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        #the entire part below i.e attention block + second resnet block is taken together as a single block which can have multiple layers i.e the below part can be in a for loop
        #attention block
        batch_size, channels, height, width=out.shape
        in_attn=out.reshape(batch_size, channels, height*width)
        in_attn=self.attention_norm(in_attn)
        in_attn=in_attn.transpose(1,2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn=out_attn.transpose(1,2).reshape(batch_size, channels, height, width)
        out = out + out_attn
        
        #second resnet block
        resnet_input=out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        return out
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample=up_sample
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )   
        self.t_emb_layers=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second=nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention_norm=nn.GroupNorm(8, out_channels)
        self.attention=nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        #upsampling is done via transposed convolution
        self.up_sample_conv=nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=4,
                                               stride=2, padding=1) if self.up_sample else nn.Identity()
        
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1) #skip connection => getting dim error here
        out = x
    
        #resnet block
        resnet_input=out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        #attention block
        batch_size, channels, height, width=out.shape
        in_attn=out.reshape(batch_size, channels, height*width)
        in_attn=self.attention_norm(in_attn)
        in_attn=in_attn.transpose(1,2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn=out_attn.transpose(1,2).reshape(batch_size, channels, height, width)
        out = out + out_attn
        
        return out 