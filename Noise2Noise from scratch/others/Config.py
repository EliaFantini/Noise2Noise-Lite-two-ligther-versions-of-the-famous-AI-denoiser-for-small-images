class Config:
    train_data_path = "./others/dataset/train_data.pkl"
    val_data_path = "./others/dataset/val_data.pkl"
    seed = 23
    num_epochs = 15
    batch_size = 100
    num_workers = 4
    device = "cpu"  # CPU is chosen,
    # change it only if you want to force a specific setting
    net = 'Network4'  # possibilities: 'Network1', 'Network2', 'Network3'
    optimizer = 'Adam'  # possibilities: 'Adam', 'SGD'
    optimizer_params = [0.001, 0.9, 0.999, 1e-8, 0, 0]  # learning rate, beta 1, beta 2, epsilon, mu, tau
    loss = 'l2'  # possibilities: 'l1'/'l2'
    normalize = False
    data_augmentation = False
    verbose = True

