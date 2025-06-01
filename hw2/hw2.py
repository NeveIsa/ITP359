# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cifar10==1.0.0",
#     "equinox==0.12.2",
#     "jax==0.6.1",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "pillow==11.2.1",
#     "seaborn==0.13.2",
#     "tensor==0.3.6",
#     "tree-math==0.2.1",
# ]
# ///

import marimo

__generated_with = "0.13.11"
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
    return cifar10, np, plt, rng, stax


@app.cell
def _(cifar10, np, plt):
    # Train data
    _count = 0
    X_train, y_train = [],[]
    for _image, _label in cifar10.data_batch_generator():
        _image # numpy array of an image, which is of shape 32 x 32 x 3
        _count += 1

        if _count <= 100:
            plt.subplot(10,10,_count)
            plt.imshow(_image)
            plt.axis("off")

        X_train.append(_image)
        y_train.append(_label)

    X_train = np.array(X_train)
    # X_train = X_train.transpose(0,3,1,2)

    y_train = np.array(y_train)

    plt.show()
    return (X_train,)


@app.cell
def _(cifar10, np):
    X_test, y_test = [],[]
    # Test data
    for _image, _label in cifar10.test_batch_generator():
        X_test.append(_image)
        y_test.append(_label)

    y_test = np.array(y_test)

    X_test = np.array(X_test)
    # X_test = X_test.transpose(0,3,1,2)
    X_test.shape
    return


@app.cell
def _():
    return


@app.cell
def _(X_train, rng, stax):
    # stax.Conv(out_chan, (kernel_h, kernel_w), strides=(sh, sw), padding='SAME')
    conv1_init, conv1 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')
    output_shape, params = conv1_init(rng, X_train.shape)


    conv2_init, conv2 = stax.Conv(10, (3, 3), strides=(1, 1), padding='SAME')
    return (params,)


@app.cell
def _(params):
    params
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
