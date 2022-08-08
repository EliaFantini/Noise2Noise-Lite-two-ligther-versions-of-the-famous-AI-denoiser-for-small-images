import torch
from .others.Config import Config
from .model import Model


def main():
    if Config.verbose:
        print("Running configuration:")
        config_values = vars(Config)
        config_keys= [v for v, m in vars(Config).items() if not (v.startswith('_')  or callable(m))]
        for key in config_keys:
            print(f"    {key} : {config_values[key]}")
    # fixing seed for reproducibility
    torch.manual_seed(Config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # loading the dataset
    train_noisy_input, train_noisy_target = torch.load(Config.train_data_path)
    # instantiating the model and all its components
    n2n_model = Model()
    # training
    n2n_model.train(train_noisy_input, train_noisy_target, num_epochs=Config.num_epochs)


if __name__ == '__main__':
    main()
