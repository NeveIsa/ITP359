import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping
    from jax import grad, jit
    import jax.numpy as jnp
    import jax

    import equinox as eqx

    import numpy as np

    import cifar10

    import matplotlib.pyplot as plt
    return Array, Float, cifar10, eqx, jax, jit, jnp, np, plt


@app.cell
def _(cifar10, np):
    # Train data
    X_train = []
    for _image, _label in cifar10.data_batch_generator():
        X_train.append(_image)

    # Test data
    X_test = []
    for _image, _label in cifar10.test_batch_generator():
        X_test.append(_image)

    X_train, X_test = map(np.array, [X_train,X_test])
    mean = X_train.mean(axis=(0,1,2)).reshape(1,1,1,-1)
    std = X_train.std(axis=(0,1,2)).reshape(1,1,1,-1)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    return X_test, X_train


@app.cell
def _(X_train, plt):
    plt.figure(figsize=(7,7))
    for _i in range(100):
        plt.subplot(10,10,_i+1)
        plt.imshow(X_train[_i])

    plt.show()  
    return


@app.cell
def _(X_test, X_train, np):
    X_train_noise = X_train + 0.5*np.random.randn(*X_train.shape)
    X_test_noise = X_test + 0.5*np.random.randn(*X_test.shape)
    return (X_train_noise,)


@app.cell
def _(X_train_noise, plt):
    plt.figure(figsize=(7,7))
    for _i in range(100):
        plt.subplot(10,10,_i+1)
        plt.imshow(X_train_noise[_i])

    plt.show()  
    return


@app.cell
def _(Array, Float, eqx, jax):
    class Encoder(eqx.Module):
        layers : list
        def __init__(self,key):
            keys = jax.random.split(key, 4)

            self.layers = [

                eqx.nn.Conv2d(3,10,kernel_size=4,key=keys[0]),
                jax.nn.relu,
                eqx.nn.MaxPool2d(kernel_size=2),

                eqx.nn.Conv2d(10,10,kernel_size=4,key=keys[1]),
                jax.nn.relu,
                eqx.nn.MaxPool2d(kernel_size=2),

                eqx.nn.Conv2d(10,10,kernel_size=4,key=keys[2]),
                jax.nn.relu,
                eqx.nn.MaxPool2d(kernel_size=2),

                eqx.nn.Conv2d(10,5,kernel_size=4,key=keys[3]),
                jax.nn.relu,
                eqx.nn.MaxPool2d(kernel_size=2), 
            ]

        def __call__(self, x: Float[Array,"1 32 32"]):
            for layer in self.layers:
                x = layer(x)
            return x
    return (Encoder,)


@app.cell
def _(Array, Float, eqx, jax, upsample_bilinear):
    class Decoder(eqx.Module):
        layers : list
        def __init__(self,key):
            keys = jax.random.split(key, 4)

            self.layers = [

                eqx.nn.Conv2d(5,10,kernel_size=8,stride=(2,2),padding=1,padding_mode="REPLICATE",key=keys[0]),
                jax.nn.relu,
                upsample_bilinear,
                eqx.nn.Conv2d(10,10,kernel_size=4,padding_mode="REPLICATE",key=keys[1]),
                jax.nn.relu,
                upsample_bilinear,

                eqx.nn.Conv2d(10,10,kernel_size=2,stride=(2,2),padding_mode="REPLICATE",key=keys[2]),
                jax.nn.relu,
                upsample_bilinear,

                eqx.nn.Conv2d(10,3,kernel_size=3,key=keys[3]),
                jax.nn.tanh,
                upsample_bilinear, 
            ]

        def __call__(self, x: Float[Array,"1 32 32"]):
            for layer in self.layers:
                x = layer(x)
            return x
    return (Decoder,)


@app.cell
def _(Decoder, Encoder, eqx, jax):
    class Autoencoder(eqx.Module):
        layers : list
        def __init__(self, key):
            enckey, deckey = jax.random.split(key)
            self.layers  = [
                Encoder(enckey),
                Decoder(deckey)
            ]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    return (Autoencoder,)


@app.cell
def _(Autoencoder, X_train, jax):
    key = jax.random.PRNGKey(0)
    model = Autoencoder(key)

    m=jax.vmap(model.layers[0])(X_train[:77].transpose(0,3,1,2))
    m.shape,jax.vmap(model.layers[1])(m).shape
    return (model,)


@app.cell
def _(X_train, jax, model):
    jax.vmap(model)(X_train[:77].transpose(0,3,1,2)).shape
    return


@app.cell
def _(X_train, X_train_noise, eqx, jax, jit, jnp, model):

    def lossfn(model, xnoisy, x):
        xnoisy = xnoisy.transpose(0,3,1,2)
        x = x.transpose(0,3,1,2)
    
        xhat = jax.vmap(model)(xnoisy)

        loss = jnp.sum((xhat - x)**2)/x.shape[0]

        return loss

    @jit
    def predict(model, x):
        x = x.transpose(0,3,1,2)
        out =  jax.vmap(model)(x)
        out.transpose(0,2,3,1)
        return out

    dlossfn = eqx.filter_grad(lossfn)

    print(lossfn(model, X_train_noise[:10], X_train[:10]))
    return dlossfn, lossfn


@app.cell
def _(X_train, X_train_noise, dlossfn, eqx, lossfn, model, np):
    import optax
    from tqdm import tqdm

    batchsize = 50
    epochs = 10

    optimizer = optax.novograd(learning_rate=1e-1)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    pbar = tqdm(range(epochs), ncols=100)

    __model = model
    for _i in pbar:
        for _b in range(0,X_train.shape[0],batchsize):
            xnoise_batch = X_train_noise[_b:_b+batchsize]
            x_batch = X_train[_b:_b+batchsize]
        
            grads = dlossfn(__model,xnoise_batch,x_batch)
            # print(x_batch.shape,xnoise_batch.shape)

            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(__model, eqx.is_array)
            )

            __model = eqx.apply_updates(__model, updates)

            if np.random.rand()>0.5:
                pbar.set_postfix({"loss":f"{lossfn(__model, xnoise_batch, x_batch):.4f}"})
    return


@app.cell
def _(mo):
    mo.md(r"""### Helpers""")
    return


@app.cell
def _(jax, jnp):
    def upsample_bilinear(x: jnp.ndarray, scale: int = 2) -> jnp.ndarray:
        """
        x:     jnp.ndarray of shape (C,H,W)
        scale: integer upsampling factor
        returns: jnp.ndarray of shape (B, H*scale, W*scale, C)
        """
        C, H, W = x.shape
        new_shape = (C, H * scale, W * scale)
        # method="bilinear" (or "bicubic"/"lanczos3", etc.)
        out =  jax.image.resize(x, new_shape, method="bilinear")

        return out
    return (upsample_bilinear,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
