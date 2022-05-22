import torch
from torch.utils.data.dataloader import DataLoader
from others.Config import Config
from others.nets import unet, unet2, unet3, DeepLabV3
from others.dataset import Dataset
import torchvision
import time  # TODO: remove this


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        if Config.net == 'Unet':
            self.net = unet.UNet()
        elif Config.net == 'Unet2':
            self.net = unet2.UNet()
        elif Config.net == 'Unet3':
            self.net = unet3.UNet()
        elif Config.net == 'DeepLab':
            self.net = DeepLabV3.createDeepLabv3()
        else:
            raise ValueError("Invalid net value. Accepted (as string): 'Unet' ")
        if Config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=Config.optimizer_params[0],
                                              betas=Config.optimizer_params[1:3], eps=Config.optimizer_params[3])
        else:
            raise ValueError("Invalid optimizer value. Accepted (as string): 'Adam' ")
        if Config.loss == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif Config.loss == 'l2':
            self.criterion = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss value. Accepted (as string): 'l1'/'l2' ")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=2,
                                                                    factor=0.5, verbose=True)
        if Config.device is not None:
            self.device = Config.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.criterion.to(self.device)

    def load_pretrained_model(self) -> None:
        best_model__state_dict = torch.load('bestmodel.pth')
        self.load_state_dict(best_model__state_dict)

    def train(self, train_input, train_target, num_epochs) -> None:
        training_loader = DataLoader(Dataset(train_input, train_target, Config), batch_size=Config.batch_size, shuffle=True,
                                     num_workers=Config.num_workers)
        val_noisy_x, val_clean_y = torch.load(Config.val_data_path)
        validation_loader = DataLoader(Dataset(val_noisy_x, val_clean_y, Config), batch_size=Config.batch_size, shuffle=False,
                                       num_workers=Config.num_workers)
        epochs_losses = []
        validation_losses = []
        validation_psnr = []
        min_loss = 9999
        if Config.verbose:
            print("Training started.")

        start = time.time()
        for epoch in range(num_epochs):
            # Training
            self.net.train(mode=True)
            epoch_loss = 0.0
            for batch_input, batch_target in training_loader:
                batch_input = batch_input.to(self.device)
                batch_target = batch_target.to(self.device)
                self.optimizer.zero_grad()
                batch_output = self.net(batch_input)
                loss = self.criterion(batch_output, batch_target)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            epochs_losses.append(epoch_loss / len(training_loader))

            # Validation
            self.net.eval()
            with torch.no_grad():
                validation_loss = 0
                psnr = 0
                for batch_input, batch_target in validation_loader:
                    if batch_input.shape[0]>1:
                        ids = torch.randperm(batch_input.shape[0])
                        sep = int(batch_input.shape[0] * 0.5)
                        sep_ids = ids[:sep]
                        batch_input = batch_input[sep_ids]
                        batch_target = batch_target[sep_ids]
                    batch_input = batch_input.to(self.device)
                    batch_target = batch_target.to(self.device)
                    batch_outputs = self.net(batch_input)
                    validation_loss += self.criterion(batch_outputs, batch_target).data.item()
                    #psnr += -10*torch.log10( torch.mean((batch_outputs - batch_target)**2)+10**-8)
                    #psnr += 10 * torch.log10(1 / torch.nn.functional.mse_loss(batch_outputs, batch_target))
                    psnr += 20*torch.log10(torch.tensor(1.0))-10*torch.log10(((batch_outputs - batch_target)**2).mean((1, 2, 3))).mean()
                validation_losses.append(validation_loss/len(validation_loader))
                validation_psnr.append(psnr / len(validation_loader))

            if Config.verbose:
                print(
                    f'Epoch: {epoch + 1}/{num_epochs} |train loss: {epochs_losses[-1]:.4f} |test loss: {validation_losses[-1]:.4f} |psnr(dB): {validation_psnr[-1]:.4f}')
            self.scheduler.step(validation_losses[-1])
            if validation_losses[-1] <= min_loss:
                torch.save(self.state_dict(), f'bestmodel.pth')
                min_loss = validation_losses[-1]

        if Config.verbose:
            print("Training finished. Best model saved.")
        elapsed_time = time.time()-start
        return epochs_losses, validation_losses, validation_psnr, elapsed_time

    def predict(self, test_input) -> torch.Tensor:
        source = test_input.div(255)
        if Config.normalize:
            mean_c1 = source[:, :, :, 0].float().mean()
            mean_c2 = source[:, :, :, 1].float().mean()
            mean_c3 = source[:, :, :, 2].float().mean()
            std_c1 = source[:, :, :, 0].float().std()
            std_c2 = source[:, :, :, 1].float().std()
            std_c3 = source[:, :, :, 2].float().std()
            transform = torchvision.transforms.Compose([torchvision.transforms.Normalize((mean_c1,mean_c2,mean_c3), (std_c1, std_c2, std_c3))])
            source = transform(source)
        source = source.to(self.device)
        return self.net(source).mul(255).to('cpu')


