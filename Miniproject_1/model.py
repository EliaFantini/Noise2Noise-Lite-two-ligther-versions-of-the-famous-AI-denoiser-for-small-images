import torch
from torch.utils.data.dataloader import DataLoader
from others.Config import Config
from others.nets.unet import UNet
from others.dataset import Dataset


class Model:
    def __init__(self) -> None:
        if Config.net == 'Unet':
            self.net = UNet()
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=Config.num_epochs / 4,
                                                                    factor=0.5)
        if Config.device is not None:
            self.device = Config.device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.to(self.device)
        self.criterion.to(self.device)

    def load_pretrained_model(self) -> None:
        ## This loads the parameters saved in bestmodel .pth into the model
        pass

    def train(self, train_input, train_target, num_epochs) -> None:
        training_loader = DataLoader(Dataset(train_input, train_target), batch_size=Config.batch_size, shuffle=True,
                                     num_workers=Config.num_workers)
        val_noisy_x, val_clean_y = torch.load(Config.val_data_path)
        validation_loader = DataLoader(Dataset(val_noisy_x, val_clean_y), batch_size=Config.batch_size, shuffle=False,
                                       num_workers=Config.num_workers)
        epochs_losses = []
        validation_losses = []
        validation_psnr = []
        if Config.verbose:
            print("Training started.")
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

        # TODO:save best model
        if Config.verbose:
            print("Training finished. Best model saved.")



    def predict(self, test_input) -> torch.Tensor:
        source = test_input.div(255).to(self.device)
        return self.net(source).mul(255).to('cpu')
