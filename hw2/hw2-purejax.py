# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cifar10==1.0.0",
#     "equinox==0.12.2",
#     "jax==0.6.1",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "openai==1.82.1",
#     "optax==0.2.4",
#     "pillow==11.2.1",
#     "seaborn==0.13.2",
#     "tensor==0.3.6",
#     "tqdm==4.67.1",
#     "tree-math==0.2.1",
# ]
# ///

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from jax import grad, jit
    import jax.numpy as jnp
    import jax
    from jax.scipy.signal import convolve2d
    from jax.example_libraries import stax
    import jax.nn as jnn

    import jax.random as random
    rng = random.PRNGKey(0)

    import numpy as np

    from cifar10_web import cifar10

    import seaborn as sns
    import matplotlib.pyplot as plt

    import equinox as eqx

    from tqdm.autonotebook import tqdm

    import optax
    return cifar10, grad, jax, jit, jnp, mo, np, optax, plt, rng, stax, tqdm


@app.cell
def _(cifar10, np):
    X_train, train_labels, X_test, test_labels = cifar10(path=None)
    X_test = X_test.reshape(X_test.shape[0],3,32,32).transpose(0,2,3,1)
    X_train = X_train.reshape(X_train.shape[0],3,32,32).transpose(0,2,3,1)


    X_train, X_test = map(np.array, [X_train,X_test])
    mean = X_train.mean(axis=(0,1,2)).reshape(1,1,1,-1)
    std = X_train.std(axis=(0,1,2)).reshape(1,1,1,-1)

    X_train = (X_train - mean)/std
    X_test = (X_test - mean)/std
    return X_test, X_train, mean, std


@app.cell
def _(X_train, plt):
    plt.figure(figsize=(7,7))
    for _i in range(100):
        plt.subplot(10,10,_i+1)
        _img = X_train[_i] - X_train[_i].min()
        plt.imshow(_img/_img.max())

    plt.show()  
    return


@app.cell
def _(X_test, X_train, np):
    X_train_noisy = X_train + 0.5*np.random.randn(*X_train.shape)
    X_test_noisy = X_test + 0.5*np.random.randn(*X_test.shape)
    return X_test_noisy, X_train_noisy


@app.cell
def _(X_train_noisy, plt):
    plt.figure(figsize=(7,7))
    for _i in range(100):
        plt.subplot(10,10,_i+1)
        _img = X_train_noisy[_i] - X_train_noisy[_i].min()
        plt.imshow(_img/_img.max())


    plt.show()  
    return


@app.cell
def _(
    conv1,
    conv2,
    conv3,
    conv4,
    conv5,
    conv6,
    conv7,
    conv8,
    conv9,
    grad,
    jax,
    jit,
    jnp,
    mpool,
    upsample2d,
):
    @jit
    def encoder(params, images):
        c1p,c2p,c3p,c4p = params 

        # Encoder
        out = conv1(c1p, images);out = jax.nn.relu(out); out = mpool(None,out)
        out = conv2(c2p,out);out = jax.nn.relu(out);out = mpool(None,out)
        out = conv3(c3p,out);out = jax.nn.relu(out);out = mpool(None,out)
        out = conv4(c4p,out);out = jax.nn.relu(out);out = mpool(None,out)


        return out

    @jit
    def decoder(params, representations):
        c5p,c6p,c7p,c8p,c9p = params

        # # Decoder
        out = conv5(c5p, representations);
        out = jax.nn.relu(out); 
        out = upsample2d(out, scale=2)

        out = conv6(c6p, out);
        out = jax.nn.relu(out); 
        out = upsample2d(out, scale=2)


        out = conv7(c7p, out); 
        out = jax.nn.relu(out); 
        out = upsample2d(out, scale=2)

        out = conv8(c8p, out); 
        out = jax.nn.relu(out); 
        out = upsample2d(out, scale=2)
    # 
        out = conv9(c9p, out);
        # out = jax.nn.tanh(out) * std *3



        return out

    @jit
    def nnet(params, images):
        representations = encoder(params[:4], images)
        reconstructions = decoder(params[4:], representations)
        return reconstructions

    @jit
    def lossfn(params, noisy_images, images):
        images_hat = nnet(params, noisy_images)
        loss = jnp.linalg.norm(images_hat - images)**2
        mse = loss/images.shape[0]
        return mse

    dlossfn = jit(grad(lossfn))


    return decoder, dlossfn, encoder, lossfn, nnet


@app.cell
def _(
    X_train,
    X_train_noisy,
    conv1params,
    conv2params,
    conv3params,
    conv4params,
    conv5params,
    conv6params,
    conv7params,
    conv8params,
    conv9params,
    dlossfn,
    get_batch_ids,
    lossfn,
    np,
    optax,
    tqdm,
):
    params = [conv1params, conv2params, conv3params, conv4params, conv5params, conv6params, conv7params, conv8params, conv9params]

    lr = 5e-4


    optimizerfn = optax.amsgrad
    optimizer = optimizerfn(learning_rate=lr)


    # # Exponential decay of the learning rate.
    # scheduler = optax.exponential_decay(
    #     init_value=lr,
    #     transition_steps=1000,
    #     decay_rate=0.999)

    # # Combining gradient transforms using `optax.chain`.
    # optimizer = optax.chain(
    #     # optax.clip_by_global_norm(1.0),  # Clip by the gradient by the global norm.
    #     optax.scale_by_adam(),  # Use the updates from adam.
    #     optax.scale_by_schedule(scheduler),  # Use the learning rate from the scheduler.
    #     # Scale updates by -1 since optax.apply_updates is additive and we want to descend on the loss.
    #     optax.scale(-1.0)
    # )

    opt_state = optimizer.init(params)

    batchsize = 500
    epochs = 20
    pbar = tqdm(range(epochs), ncols=100)
    for _ in pbar:

        for bids in get_batch_ids(X_train.shape[0], batch_size=batchsize):
            _Xnoisy, _X = X_train_noisy[bids], X_train[bids]

            grads = dlossfn(params, _Xnoisy, _X)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if np.random.rand()>0.25:
                _loss = lossfn(params, _Xnoisy, _X)
                pbar.set_postfix({"loss":f"{_loss:.4f}", "optimfn":optimizerfn.__name__, "batchsize":f"{batchsize}"})
    return (params,)


@app.cell
def _(X_test, X_test_noisy, mean, nnet, np, params, plt, std):
    imgid = np.random.randint(300)

    plt.subplot(1,3,1)
    _img = X_test_noisy[imgid] - X_test_noisy[imgid].min()
    _img = _img / _img.max()
    plt.imshow(_img)

    plt.subplot(1,3,2)
    _img = nnet(params, X_test_noisy[imgid:imgid+1])[0]
    _img = _img*std[0] + mean[0]
    _img = _img / _img.max()
    plt.imshow(_img)


    plt.subplot(1,3,3)
    _img = X_test[imgid]
    _img = _img*std[0] + mean[0]
    _img = _img / _img.max()
    plt.imshow( _img)
    return


@app.cell
def _(mo):
    mo.md(r"""### HELPERS""")
    return


@app.cell
def _(jax, jnp):
    # 1) Nearest-neighbor 2× upsampling in NHWC
    # def upsample2d(x, scale=2):
    #     # x: (batch, H, W, channels)
    #     y = jnp.repeat(x, repeats=scale, axis=1)  # repeat rows
    #     y = jnp.repeat(y, repeats=scale, axis=2)  # repeat cols
    #     return y

    def upsample2d(x: jnp.ndarray, scale: int = 2) -> jnp.ndarray:
        """
        x:     jnp.ndarray of shape (C,H,W)
        scale: integer upsampling factor
        returns: jnp.ndarray of shape (B, H*scale, W*scale, C)
        """
        B,H, W, C = x.shape
        new_shape = (B, H * scale, W * scale, C)
        # method="bilinear" (or "bicubic"/"lanczos3", etc.)
        out =  jax.image.resize(x, new_shape, method="bilinear")

        return out

    # upsample2d = jax.vmap(__upsample2d)
    return (upsample2d,)


@app.cell
def _(X_test, np, rng, stax, upsample2d):


    mpool_init,mpool = stax.MaxPool(
        window_shape=(2, 2),
        strides=(2, 2),
        padding='VALID'   # or 'SAME'
    )

    OUTCHANNELS = 100

    # stax.Conv(out_chan, (kernel_h, kernel_w), strides=(sh, sw), padding='SAME')
    conv1_init, conv1 = stax.Conv(OUTCHANNELS, (10, 10), strides=(1, 1), padding='SAME')
    output_shape, conv1params = conv1_init(rng, X_test.shape)
    out_shape, _params = mpool_init(rng, output_shape)

    conv2_init, conv2 = stax.Conv(OUTCHANNELS, (7, 7), strides=(1, 1), padding='SAME')
    out_shape, conv2params = conv2_init(rng, output_shape)
    out_shape, _params = mpool_init(rng, output_shape)

    conv3_init, conv3 = stax.Conv(OUTCHANNELS, (5, 5), strides=(1, 1), padding='SAME')
    out_shape, conv3params = conv3_init(rng, output_shape)
    out_shape, _params = mpool_init(rng, output_shape)


    conv4_init, conv4 = stax.Conv(OUTCHANNELS, (4, 4), strides=(1, 1), padding='SAME')
    out_shape, conv4params = conv4_init(rng, output_shape)
    out_shape, _params = mpool_init(rng, output_shape)


    ### Decoder - Conv and Up Sample

    conv5_init, conv5 = stax.ConvTranspose(OUTCHANNELS, (4, 4), strides=(1, 1), padding='SAME')
    out_shape, conv5params = conv5_init(rng, output_shape)
    out_shape = upsample2d(np.random.rand(*out_shape)[:10]).shape

    conv6_init, conv6 = stax.ConvTranspose(OUTCHANNELS, (5, 5), strides=(1, 1), padding='SAME')
    out_shape, conv6params = conv6_init(rng, output_shape)
    out_shape = upsample2d(np.random.rand(*out_shape)[:10]).shape

    conv7_init, conv7 = stax.ConvTranspose(OUTCHANNELS, (5, 5), strides=(1, 1), padding='SAME')
    out_shape, conv7params = conv7_init(rng, output_shape)
    out_shape = upsample2d(np.random.rand(*out_shape)[:10]).shape

    conv8_init, conv8 = stax.ConvTranspose(OUTCHANNELS, (4, 4), strides=(1, 1), padding='SAME')
    out_shape, conv8params = conv8_init(rng, output_shape)
    out_shape = upsample2d(np.random.rand(*out_shape)[:10]).shape


    conv9_init, conv9 = stax.ConvTranspose(3, (3, 3), strides=(1, 1), padding='SAME')
    out_shape, conv9params = conv9_init(rng, output_shape)


    print(out_shape)

    return (
        conv1,
        conv1params,
        conv2,
        conv2params,
        conv3,
        conv3params,
        conv4,
        conv4params,
        conv5,
        conv5params,
        conv6,
        conv6params,
        conv7,
        conv7params,
        conv8,
        conv8params,
        conv9,
        conv9params,
        mpool,
    )


@app.cell
def _(
    X_test,
    conv1params,
    conv2params,
    conv3params,
    conv4params,
    conv5params,
    conv6params,
    conv7params,
    conv8params,
    conv9params,
    decoder,
    encoder,
):
    _out = encoder([conv1params, conv2params, conv3params, conv4params],X_test[:19])
    _out2 = decoder([conv5params, conv6params,conv7params,conv8params, conv9params],_out)
    _out.shape, _out2.shape
    return


@app.cell
def _(np):
    def get_batch_ids(n, batch_size):
        ids = np.arange(n)
        np.random.shuffle(ids)

        for start in range(0, n, batch_size):
            batch_ids = ids[start : start + batch_size]
            yield batch_ids
    return (get_batch_ids,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
