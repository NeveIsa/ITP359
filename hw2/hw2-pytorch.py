import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from cifar10_web import cifar10
    from matplotlib import pyplot as plt

    import numpy as np

    import torch.nn as nn
    import torch
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    # from kornia.filters import UnsharpMask


    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return (
        DataLoader,
        TensorDataset,
        cifar10,
        device,
        nn,
        np,
        optim,
        plt,
        torch,
        tqdm,
    )


@app.cell
def _(cifar10, np, torch):
    X_train, train_labels, X_test, test_labels = cifar10(path=None)

    X_train = X_train.reshape(-1,3,32,32)
    X_test = X_test.reshape(-1,3,32,32)

    mean = X_train.mean(axis=(0,2,3)).reshape(1,3,1,1)
    std = X_train.std(axis=(0,2,3)).reshape(1,3,1,1)

    X_train = (X_train - mean) / std 
    X_test = (X_test - mean) /std 

    X_train_noisy = X_train + np.random.randn(*X_train.shape)*0.2
    X_test_noisy = X_test + np.random.randn(*X_test.shape)*0.2

    X_train, X_test, X_train_noisy, X_test_noisy = map(torch.tensor, [X_train, X_test, X_train_noisy, X_test_noisy])

    # X_train, X_test, X_train_noisy, X_test_noisy = map(lambda x:x.to(device), [X_train, X_test, X_train_noisy, X_test_noisy])

    return X_test, X_test_noisy, X_train, X_train_noisy, mean, std


@app.cell
def _(X_test_noisy, image_format, plt):
    # _imgs = image_format(model(X_test[:100].to(device)))

    plt.figure(figsize=(7,7))
    for _imgid in range(100):
        plt.subplot(10,10,_imgid+1)    

        # plt.subplot(1,3,1)
        # plt.imshow(image_format(X_test[_imgid].reshape(1,3,32,32))[0])

        # plt.subplot(1,3,2)
        plt.imshow(image_format(X_test_noisy[_imgid].reshape(1,3,32,32))[0])

        # plt.subplot(10,3,3)
        # plt.imshow(imgs[_imgid]/imgs[_imgid].max())
    plt.show()
    return


@app.cell
def _(X_test, device, nn, torch):
    encoder = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=100, kernel_size=2),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=100, out_channels=100, kernel_size=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=100, out_channels=100, kernel_size=2),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=100, out_channels=100, kernel_size=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten(),
        nn.Linear(100*6*6, 100*6*6),
        nn.Unflatten(1,(100,6,6))



    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(in_channels=100, out_channels=100, kernel_size=2),
        nn.ReLU(),
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),


        nn.ConvTranspose2d(in_channels=100, out_channels=50, kernel_size=2),
        nn.ReLU(),
        # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=3),
        nn.ReLU(),


        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=3),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

        nn.ReLU(),

        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),


        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),
        # UnsharpMask(kernel_size=5, sigma=(1.0, 1.0)),


        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels=50, out_channels=50, kernel_size=2),
        nn.ReLU(),

        nn.ConvTranspose2d(in_channels=50, out_channels=3, kernel_size=2),
        # UnsharpMask(kernel_size=5, sigma=(1.0, 1.0))


        # nn.Flatten(),
        # nn.Linear(3*32*32, 3*32*32),
        # nn.Unflatten(1,(3,32,32)),
        nn.Tanh()
    )


    model = nn.Sequential(encoder,decoder).to(device)
    model = torch.compile(model)
    print(encoder(X_test[:17].to(device)).shape)
    print(model(X_test[:17].to(device)).shape)
    return (model,)


@app.cell
def _(DataLoader, TensorDataset, X_train, X_train_noisy, model, nn, optim):
    lossfn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, amsgrad=True,decoupled_weight_decay=1e-3)

    dataset = TensorDataset(X_train_noisy, X_train)
    train_loader = DataLoader(dataset, batch_size=700, shuffle=True)
    return lossfn, optimizer, train_loader


@app.cell
def _(device, lossfn, model, optimizer, tqdm, train_loader):
    epochs = 30
    pbar = tqdm(range(epochs), ncols=90, colour="green")

    for _ in pbar:
        for Xtnb, Xtb in train_loader:
            Xtnb, Xtb = Xtnb.to(device), Xtb.to(device)
            X_hat = model(Xtb)
            loss = lossfn(X_hat, Xtb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix_str(f"loss: {loss*100:.1f}%")

    return


@app.cell
def _(device, lossfn, model, optim, tqdm, train_loader):
    optimizer2 = optim.Adam(model.parameters(), lr=1e-4,amsgrad=True)#,weight_decay=1e-5)
    epochs2 = 30
    pbar2 = tqdm(range(epochs2), ncols=90, colour="green")

    for _ in pbar2:
        for Xtnb2, Xtb2 in train_loader:
            Xtnb2, Xtb2 = Xtnb2.to(device), Xtb2.to(device)
            X_hat2 = model(Xtb2)
            loss2 = lossfn(X_hat2, Xtb2)
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            pbar2.set_postfix_str(f"loss: {loss2*100:.1f}%")

    return


@app.cell
def _(X_test, X_test_noisy, device, image_format, model, np, plt):
    img = image_format(model(X_test[:100].to(device)))

    _imgid = np.random.randint(100)

    plt.subplot(1,3,1)
    plt.imshow(image_format(X_test[_imgid].reshape(1,3,32,32))[0])

    plt.subplot(1,3,2)
    plt.imshow(image_format(X_test_noisy[_imgid].reshape(1,3,32,32))[0])

    plt.subplot(1,3,3)
    plt.imshow(img[_imgid]/img[_imgid].max())

    return


@app.cell
def _(mean, std):
    def image_format(img):
        img  = img.detach().cpu().numpy()
        img = img.transpose(0,2,3,1)
        img = img * std.reshape(1,1,1,3)
        img = img + mean.reshape(1,1,1,3)
        return img
    return (image_format,)


@app.cell
def _(X_test, device, image_format, model, plt):
    imgs = image_format(model(X_test[:100].to(device)))

    plt.figure(figsize=(7,7))
    for _imgid in range(100):
        plt.subplot(10,10,_imgid+1)    

        # plt.subplot(1,3,1)
        # plt.imshow(image_format(X_test[_imgid].reshape(1,3,32,32))[0])

        # plt.subplot(1,3,2)
        # plt.imshow(image_format(X_test_noisy[_imgid].reshape(1,3,32,32))[0])

        # plt.subplot(10,3,3)
        plt.imshow(imgs[_imgid]/imgs[_imgid].max())
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
