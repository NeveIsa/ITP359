# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "altair==5.5.0",
#     "jax==0.6.1",
#     "jaxopt==0.8.5",
#     "marimo",
#     "matplotlib==3.10.3",
#     "numpy==2.2.6",
#     "openai==1.82.1",
#     "optax==0.2.4",
#     "plotly==6.1.2",
#     "polars==1.30.0",
#     "python-dateutil==2.9.0.post0",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
#     "tqdm==4.67.1",
# ]
# ///

import marimo

__generated_with = "0.13.11"
app = marimo.App(width="medium")


@app.cell
def _():
    from jax import grad, jit, value_and_grad
    import jax.numpy as jnp
    import jax.nn

    import optax

    from sklearn.model_selection import train_test_split

    import polars as pl

    import seaborn as sns
    import matplotlib.pyplot as plt

    from dateutil.parser import parse as dtparse
    from zoneinfo import ZoneInfo

    import marimo as mo
    import altair as alt

    from tqdm import tqdm
    import numpy as np
    return (
        ZoneInfo,
        alt,
        dtparse,
        jax,
        jit,
        jnp,
        mo,
        np,
        optax,
        pl,
        plt,
        sns,
        tqdm,
        train_test_split,
        value_and_grad,
    )


@app.cell
def _(pl, string2datetime):
    data = pl.read_csv("taxifares.csv")
    data = data.filter((data["pickup_latitude"] != 0) & 
                                (data["pickup_longitude"] != 0) & 
                                (data["dropoff_latitude"] != 0) & 
                                (data["dropoff_longitude"] != 0))


    #### Calculate L1 and L2 distance
    _cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
    picklat, picklong, droplat, droplong = list(map(lambda x: data[x].to_numpy() , _cols))

    a = (picklat - droplat)* 69.0
    b = (picklong - droplong)* 54.6


    l2 = (a**2 + b**2)**0.5 # l2 distance
    l1 = abs(a) + abs(b)  # l1 distance

    data = data.with_columns([pl.Series("l1",l1),pl.Series("l2",l2)])
    data = data.filter(data["l2"]<40) # filter the sane values, the rest are anomalous


    #### Set day of Weekday and Time
    dayofweek, time_of_day, hours_passed = list(zip(*list(map(string2datetime, data["pickup_datetime"].to_list()))))
    data = data.with_columns([pl.Series("day",dayofweek),pl.Series("time_of_day",time_of_day),pl.Series("time",hours_passed)])

    #### Drop Latitude and Logitude columns
    data = data.drop(["pickup_latitude","pickup_longitude","dropoff_latitude","dropoff_longitude","pickup_datetime"])
    return (data,)


@app.cell
def _(data):
    target = data["fare_amount"].to_numpy();target
    return (target,)


@app.cell
def _(data):
    data.head()
    return


@app.cell
def _(alt, data, mo, plt, sns):

    chart = mo.ui.altair_chart(alt.Chart(data).mark_point().encode(
        x='l2',
        y='fare_amount',
    ))
    # plt.xlim(0,100)
    plt.figure(figsize=(12, 4))
    sns.scatterplot(data, x='l2', y='fare_amount', marker='.', hue='day', palette="mako")

    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(data, target, train_test_split):
    selected_data = data.select(["l1", "l2", "day", "time"])
    X = selected_data.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    return X, X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### JAX Neural Net Model from Scratch""")
    return


@app.cell
def _(jax, jit, jnp, value_and_grad):
    @jit
    def nnet(params, X, actfn=jax.nn.sigmoid):
        for p in params[:-1]:
            X = actfn(X @ p)
        X  = X @ p[-1]
        return X

    @jit
    def lossfn(params, X, y):
        yhat = nnet(params,X)
        error = jnp.linalg.norm(yhat - y)/y.shape[0]
        return error


    loss_val_and_grad_fn = jit(value_and_grad(lossfn))
    return loss_val_and_grad_fn, lossfn, nnet


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Parameter Initialization and Training via gradient descent""")
    return


@app.cell
def _(mo):
    lr = mo.ui.slider(5e-2,5,1e-1, show_value=True, label="lr")
    epochs = mo.ui.slider(300,2000,show_value=True,label='epoch')
    lr,epochs
    return epochs, lr


@app.cell
def _():
    return


@app.cell
def _(X, X_train, epochs, loss_val_and_grad_fn, lr, np, optax, tqdm, y_train):
    def adam_train(): 
        start_lr = lr.value
        _epochs = epochs.value
        momentum_weight = 0.7
        # 2 hidden layers, one last dense layer
        params = [np.random.randn(X.shape[1],100), np.random.randn(100,100), np.random.randn(100,1)]
    
        optimizer = optax.adam(learning_rate=start_lr, b1=momentum_weight)
        opt_state = optimizer.init(params)
    
        pbar = tqdm(range(_epochs))
        for step in pbar:
            loss, grads = loss_val_and_grad_fn(params, X_train, y_train)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
    
            pbar.set_postfix({"loss":f"{loss:.4f}"})

        return params
    return (adam_train,)


@app.cell
def _(X, X_train, epochs, loss_val_and_grad_fn, lr, np, tqdm, y_train):
    def gd_train():
        start_lr = lr.value*2
        _epochs = epochs.value*5
        momentum_weight = 0.9
        # 2 hidden layers, one last dense layer
        params = [np.random.randn(X.shape[1],100), np.random.randn(100,100), np.random.randn(100,1)]
    
        momentum = [np.zeros_like(p) for p in params]
    
        pbar = tqdm(range(_epochs))
    
        for step in pbar:
            loss, grads = loss_val_and_grad_fn(params, X_train, y_train)
            for i in range(len(params)):
                params[i] = params[i] - (start_lr  * grads[i]) - momentum[i]
                momentum[i] = grads[i]*momentum_weight + momentum[i]*(1-momentum_weight)
                start_lr = min((1+5e-3)*start_lr,10)
                # momentum_weight = 0.9999*momentum_weight
    
            pbar.set_postfix({"loss":f"{loss:.4f}","lr":f"{start_lr:.4f}"})

        return params
    return (gd_train,)


@app.cell
def _(X_test, gd_train, lossfn, nnet, y_test):
    params_gd = gd_train()
    test_err_gd = lossfn(params_gd,X_test,y_test); 
    print("gd test error:",test_err_gd)
    print("gd pred:", nnet(params_gd, X_test))
    print("trueval:", y_test)
    return


@app.cell
def _(X_test, adam_train, lossfn, nnet, y_test):
    params_adam = adam_train()
    test_err_adam = lossfn(params_adam,X_test,y_test); 
    print("adam test error:",test_err_adam)
    print("adam pred:", nnet(params_adam, X_test))
    print("trueval:  ", y_test)
    return


@app.cell
def _(X_test, nnet, params):
    nnet(params, X_test)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Helpers""")
    return


@app.cell
def _(ZoneInfo, dtparse):
    def string2datetime(string):
        dt_naive = dtparse(string)
        dt_ny = dt_naive.replace(tzinfo=ZoneInfo("America/New_York"))
        day_of_week = dt_ny.weekday()
        time_of_day = dt_ny.strftime("%H:%M:%S")
        minutes_passed_since_today = dt_ny.hour  + dt_ny.minute/60
        return (day_of_week, time_of_day, minutes_passed_since_today)
    return (string2datetime,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
