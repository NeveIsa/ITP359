# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cifar10==1.0.0",
#     "equinox==0.12.2",
#     "jax==0.6.1",
#     "marimo",
#     "matplotlib==3.10.3",
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

    import cifar10

    import seaborn as sns
    import matplotlib.pyplot as plt

    import equinox as eqx
    return cifar10, eqx, jax, jnp, plt


@app.cell
def _(cifar10, plt):
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

    plt.show()
    return


@app.cell
def _(cifar10):
    X_test, y_test = [],[]
    # Test data
    for _image, _label in cifar10.test_batch_generator():
        X_test.append(_image)
        y_test.append(_label)
    return


@app.cell
def _(eqx, jax, jnp):
    conv = eqx.nn.Conv2d(
        in_channels=3,
        out_channels=8,
        kernel_size=5,
        stride=1,
        padding=2,  # SAME padding for kernel_size=5
        key=jax.random.PRNGKey(0)
    )

    x = jnp.ones((3, 32, 32))  # single image, 3 channels
    y = conv(x)
    print(y.shape)  # → (8, 32, 32)
    return (conv,)


@app.cell
def _(conv, jax):
    jax.vmap(conv, ())
    return


if __name__ == "__main__":
    app.run()
