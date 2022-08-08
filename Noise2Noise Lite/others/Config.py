class Config:
    train_data_path = "./others/dataset/train_data.pkl"
    val_data_path = "./others/dataset/val_data.pkl"
    seed = 23
    num_epochs = 20
    batch_size = 100
    num_workers = 4
    device = None  # It is assigned automatically on runtime based on pc's hardware,
    # change it only if you want to force a specific setting
    net = 'Unet'  # possibilities: 'Unet','Unet2' Unet our architecture with weight sharing,
    # Unet2 is the original Noise2Noise architecture
    optimizer = 'Adam'  # possibilities: 'Adam'
    optimizer_params = [0.006, 0.9, 0.999, 1e-8]  # learning rate, beta 1, beta 2, epsilon
    loss = 'l2'  # possibilities: 'l1'/'l2'
    normalize = True
    data_augmentation = True
    verbose = True

