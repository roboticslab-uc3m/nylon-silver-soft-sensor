import os

from catasta.models import (
    ApproximateGPRegressor,
    TransformerRegressor,
    MambaRegressor,
    FeedforwardRegressor,
)
from catasta.datasets import RegressionDataset
from catasta.scaffolds import RegressionScaffold
from catasta.dataclasses import RegressionTrainInfo, RegressionEvalInfo
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
    Custom,
)

DATA_DIR = "data/nylon_elastic_wire/paper/"


def train(model, context_length: int, dataset_name: str) -> None:
    print(f"Training {model.__class__.__name__} on {dataset_name} dataset")

    input_trasnsformations = [
        Custom(lambda x: x[1_200_000:1_500_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
        WindowSliding(window_size=context_length, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[1_200_000:1_500_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
        Slicing(amount=context_length-1, end="left"),
    ]

    dataset_path: str = os.path.join(DATA_DIR, dataset_name)
    signals = os.listdir(dataset_path)

    for signal in signals:
        root: str = os.path.join(dataset_path, signal)
        n_files: int = len(os.listdir(root))

        train_split: float = (n_files - 1) / n_files
        dataset = RegressionDataset(
            root=root,
            input_transformations=input_trasnsformations,
            output_transformations=output_trasnsformations,
            splits=(train_split, 0.0, 1 - train_split) if isinstance(model, ApproximateGPRegressor) else (train_split, 1 - train_split, 0.0),
        )

        scaffold = RegressionScaffold(
            model=model,
            dataset=dataset,
            optimizer="adamw",
            loss_function="variational_elbo" if isinstance(model, ApproximateGPRegressor) else "mse",
        )

        print(f"\nTraining {signal}...")

        train_info: RegressionTrainInfo = scaffold.train(
            epochs=100,
            batch_size=256,
            lr=1e-4,
        )
        print(f"Best loss: {train_info.best_train_loss}")

        eval_info: RegressionEvalInfo = scaffold.evaluate()
        print(eval_info)

        save_path: str = f"models/{dataset_name}/{signal}/"
        scaffold.save(path=save_path)


def gp() -> None:
    strain_context_length: int = 1536
    strain_model = ApproximateGPRegressor(
        context_length=strain_context_length,
        n_inducing_points=32,
        kernel="rq",
        mean="constant",
        use_ard=True,
    )

    stress_context_length: int = 720
    stress_model = ApproximateGPRegressor(
        context_length=stress_context_length,
        n_inducing_points=128,
        kernel="rq",
        mean="constant",
        use_ard=True,
    )

    train(strain_model, strain_context_length, "strain")
    train(stress_model, stress_context_length, "stress")


def transformer() -> None:
    strain_context_length: int = 1920
    strain_model = TransformerRegressor(
        context_length=strain_context_length,
        n_patches=8,
        d_model=16,
        n_heads=4,
        n_layers=2,
        feedforward_dim=4,
        head_dim=4,
        dropout=0.1,
    )

    stress_context_length: int = 720
    stress_model = TransformerRegressor(
        context_length=stress_context_length,
        n_patches=16,
        d_model=8,
        n_heads=2,
        n_layers=4,
        feedforward_dim=2,
        head_dim=16,
        dropout=0.1,
    )

    train(strain_model, strain_context_length, "strain")
    train(stress_model, stress_context_length, "stress")


def mamba() -> None:
    strain_context_length: int = 1152
    strain_model = MambaRegressor(
        context_length=strain_context_length,
        n_patches=16,
        d_model=8,
        d_state=8,
        d_conv=2,
        expand=3,
        n_layers=2,
    )

    stress_context_length: int = 1280
    stress_model = MambaRegressor(
        context_length=stress_context_length,
        n_patches=8,
        d_model=16,
        d_state=2,
        d_conv=4,
        expand=1,
        n_layers=3,
    )

    train(strain_model, strain_context_length, "strain")
    train(stress_model, stress_context_length, "stress")


def fnn() -> None:
    strain_context_length: int = 720
    strain_model = FeedforwardRegressor(
        context_length=strain_context_length,
        hidden_dims=[64, 128, 256],
        dropout=0.1,
        use_batch_norm=True,
        use_layer_norm=False,
    )

    stress_context_length: int = 960
    stress_model = FeedforwardRegressor(
        context_length=stress_context_length,
        hidden_dims=[128, 256, 64],
        dropout=0.1,
        use_batch_norm=True,
        use_layer_norm=False,
    )

    train(strain_model, strain_context_length, "strain")
    train(stress_model, stress_context_length, "stress")


if __name__ == "__main__":
    gp()
    transformer()
    mamba()
    fnn()
