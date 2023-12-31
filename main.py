import numpy as np
import matplotlib.pyplot as plt

from src import ApproximateGPRegressor, ModelDataset, GaussianRegressorScaffold, EvalInfo, FeedforwardRegressor, RegressorScaffold


def gp() -> None:
    n_inducing_points: int = 128
    n_dim: int = 20
    dataset_root: str = "data/strain/"
    # dataset_root: str = "data/stress/"
    dataset = ModelDataset(root=dataset_root, n_dim=n_dim)
    model = ApproximateGPRegressor(n_inducing_points, n_dim, kernel="rq", mean="constant")
    scaffold = GaussianRegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(losses)
    plt.show()

    info: EvalInfo = scaffold.evaluate(plot_results=True)

    print(info)


def fnn() -> None:
    n_dim: int = 20
    dataset_root: str = "data/strain/"
    # dataset_root: str = "data/stress/"
    dataset = ModelDataset(root=dataset_root, n_dim=n_dim)
    model = FeedforwardRegressor(
        input_dim=n_dim,
        output_dim=1,
        hidden_dims=[64, 32, 16],
        dropout=0.1,
    )
    scaffold = RegressorScaffold(model, dataset)

    losses: np.ndarray = scaffold.train(
        epochs=100,
        batch_size=256,
        train_split=6/7,
        lr=1e-3,
    )

    plt.figure(figsize=(30, 20))
    plt.plot(losses)
    plt.show()

    info: EvalInfo = scaffold.evaluate(plot_results=True)

    print(info)


if __name__ == '__main__':
    gp()
    fnn()
