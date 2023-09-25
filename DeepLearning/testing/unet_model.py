""" Full assembly of the parts to form the complete network """

from unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))
    def encoder(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return [x5, [x4, x3, x2, x1]]

    def decoder(self, x, skips):

        x = self.up1(x, skips[0])
        x = self.up2(x, skips[1])
        x = self.up3(x, skips[2])
        x = self.up4(x, skips[3])
        out = self.outc(x)

        return out


    def forward(self, x, y = None, training = True):
        x = x.unsqueeze(dim=2)
        if training == True: # train with teacher forcing
            s = self.encoder(x[:, -1, :, :]) # get skips of last input
            res = []
            for t in range(x.size(1)): #timesteps in y
                if t == 0:
                    s = self.decoder(s[0], s[1])
                    res.append(s)
                    s = y[:, t, :, :].unsqueeze(dim = 1) # change with target
                if t != 0:
                    s = self.encoder(s)
                    s = self.decoder(s[0], s[1])
                    res.append(s)
                    s = y[:, t, :, :].unsqueeze(dim = 1)

            # save over time dimension
            out = torch.stack(res, dim = 1).squeeze()

        if training == False:
            s = self.encoder(x[:, -1, :, :])  # get skips of last input
            res = []
            for t in range(x.size(1)):  # timesteps in y
                if t == 0:
                    s = self.decoder(s[0], s[1])
                    res.append(s)
                if t != 0:
                    s = self.encoder(s)
                    s = self.decoder(s[0], s[1])
                    res.append(s)

            # save over time dimension
            out = torch.stack(res, dim=1).squeeze()
        return out

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
