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

    import cifar10

    import seaborn as sns
    import matplotlib.pyplot as plt

    import equinox as eqx

    from tqdm.autonotebook import tqdm

    import optax
    return cifar10, grad, jit, jnp, mo, np, optax, plt, rng, stax, tqdm


@app.cell
def _(cifar10, np, plt):
    # Train data
    _count = 0
    X_train, y_train = [],[]
    X_train_noisy = []
    for _image, _label in cifar10.data_batch_generator():
        _image # numpy array of an image, which is of shape 32 x 32 x 3
        _count += 1

        if _count <= 100:
            plt.subplot(10,10,_count)
            plt.imshow(_image)
            plt.axis("off")

        X_train.append(_image/255)

        _noisyimg = np.random.randn(*_image.shape) + _image/255
        X_train_noisy.append(_noisyimg)

        y_train.append(_label)

    X_train = np.array(X_train)
    # X_train = X_train.transpose(0,3,1,2)

    X_train_noisy = np.array(X_train_noisy)

    y_train = np.array(y_train)

    plt.show()
    return X_train, X_train_noisy


@app.cell
def _(cifar10, np):
    X_test, y_test = [],[]
    # Test data
    for _image, _label in cifar10.test_batch_generator():
        X_test.append(_image/255)
        y_test.append(_label)

    y_test = np.array(y_test)

    X_test = np.array(X_test)
    # X_test = X_test.transpose(0,3,1,2)
    X_test.shape
    return (X_test,)


@app.cell
def _(
    conv1,
    conv2,
    conv3,
    conv4,
    conv5,
    conv6,
    grad,
    jit,
    jnp,
    mpool1,
    mpool2,
    mpool3,
    upsample2d,
):
    @jit
    def encoder(params, images):
        c1p,c2p,c3p = params 

        # Encoder
        out = conv1(c1p, images);out = mpool1(None,out)
        out = conv2(c2p,out);out = mpool2(None,out)
        out = conv3(c3p,out);out = mpool3(None,out)

        return out

    @jit
    def decoder(params, representations):
        c4p,c5p,c6p = params
        # # Decoder
        out = conv4(c4p, representations); out = upsample2d(out)
        out = conv5(c5p, out); out = upsample2d(out)
        out = conv6(c6p, out); out = upsample2d(out)

        return out

    @jit
    def nnet(params, images):
        representations = encoder(params[:3], images)
        reconstructions = decoder(params[3:], representations)
        return reconstructions

    @jit
    def lossfn(params, noisy_images, images):
        images_hat = nnet(params, noisy_images)
        loss = jnp.linalg.norm(images_hat - images)**2
        mse = loss/images.shape[0]
        return mse

    dlossfn = jit(grad(lossfn))
    return dlossfn, lossfn


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
    dlossfn,
    get_batch_ids,
    lossfn,
    np,
    optax,
    tqdm,
):
    params = [conv1params, conv2params, conv3params, conv4params, conv5params, conv6params]

    start_learning_rate = 1e-2
    optimizer = optax.adam(start_learning_rate)
    opt_state = optimizer.init(params)

    epochs = 10
    pbar = tqdm(range(epochs), ncols=50)
    for _ in pbar:
        for bids in get_batch_ids(X_train.shape[0]):
            _Xnoisy, _X = X_train_noisy[bids], X_train[bids]

            grads = dlossfn(params, _Xnoisy, _X)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            if np.random.rand()>0.5:
                _loss = lossfn(params, _Xnoisy, _X)
                pbar.set_postfix({"loss":f"{_loss:.4f}"})
    return


@app.cell
def _(mo):
    mo.md(r"""### HELPERS""")
    return


@app.cell
def _(jnp):
    # 1) Nearest-neighbor 2× upsampling in NHWC
    def upsample2d(x, scale=2):
        # x: (batch, H, W, channels)
        y = jnp.repeat(x, repeats=scale, axis=1)  # repeat rows
        y = jnp.repeat(y, repeats=scale, axis=2)  # repeat cols
        return y
    return (upsample2d,)


@app.cell(hide_code=True)
def _(X_test, X_train, rng, stax, upsample2d):
    # stax.Conv(out_chan, (kernel_h, kernel_w), strides=(sh, sw), padding='SAME')
    conv1_init, conv1 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')
    output_shape, conv1params = conv1_init(rng, X_train.shape)

    mpool1_init,mpool1 = stax.MaxPool(
        window_shape=(2, 2),
        strides=(2, 2),
        padding='VALID'   # or 'SAME'
    )

    out_shape, _params = mpool1_init(rng, output_shape)

    conv2_init, conv2 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')

    out_shape, conv2params = conv2_init(rng, output_shape)

    mpool2_init,mpool2 = stax.MaxPool(
        window_shape=(2, 2),
        strides=(2, 2),
        padding='VALID'   # or 'SAME'
    )

    out_shape, _params = mpool2_init(rng, output_shape)

    conv3_init, conv3 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')

    out_shape, conv3params = conv3_init(rng, output_shape)

    mpool3_init,mpool3 = stax.MaxPool(
        window_shape=(2, 2),
        strides=(2, 2),
        padding='VALID'   # or 'SAME'
    )

    out_shape, _params = mpool3_init(rng, output_shape)


    ### Decoder - Conv and Up Sample

    conv4_init, conv4 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')
    out_shape, conv4params = conv4_init(rng, output_shape)

    out_shape = upsample2d(X_test[:2]).shape

    conv5_init, conv5 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')
    out_shape, conv5params = conv5_init(rng, output_shape)

    out_shape = upsample2d(X_test[:2]).shape

    conv6_init, conv6 = stax.Conv(3, (3, 3), strides=(1, 1), padding='SAME')
    out_shape, conv6params = conv6_init(rng, output_shape)


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
        mpool1,
        mpool2,
        mpool3,
    )


@app.cell
def _(np):
    def get_batch_ids(n, batch_size=100):
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
