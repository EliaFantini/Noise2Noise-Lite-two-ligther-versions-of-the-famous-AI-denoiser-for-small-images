class Config:
    train_data_path = "./others/dataset/train_data.pkl"
    val_data_path = "./others/dataset/val_data.pkl"
    seed = 23
    num_epochs = 10
    batch_size = 100
    num_workers = 4
    device = None  # It is assigned automatically on runtime based on pc's hardware,
    # change it only if you want to force a specific setting
    net = 'Unet'  # possibilities: 'Unet'
    optimizer = 'Adam'  # possibilities: 'Adam'
    optimizer_params = [1e-3, 0.9, 0.999, 1e-8]  # learning rate, beta 1, beta 2, epsilon
    loss = 'l2'  # possibilities: 'l1'/'l2'
    verbose = True

