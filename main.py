import numpy as np
import matplotlib.pyplot as plt

from src import ApproximateGPRegressor, ModelDataset, GaussianRegressorScaffold, EvalInfo


def main() -> None:
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


if __name__ == '__main__':
    main()
