import os
import time
import gc

import numpy as np
import pandas as pd

from catasta.models import (
    ApproximateGPRegressor,
    TransformerRegressor,
    MambaRegressor,
    FeedforwardRegressor,
)
from catasta.dataclasses import RegressionPrediction, RegressionEvalInfo
from catasta.archways import RegressionArchway
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
    Custom,
)

from vclog import Logger


def inference(model, context_length: int, dataset_name: str) -> None:
    data_path: str = f"data/nylon_elastic_wire/{dataset_name}/"
    files: list[str] = os.listdir(data_path)
    files.sort()

    input_trasnsformations = [
        Custom(lambda x: x[800_000:950_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
        WindowSliding(window_size=context_length, stride=1),
    ]
    output_trasnsformations = [
        Custom(lambda x: x[800_000:950_000]),
        Normalization("minmax"),
        Decimation(decimation_factor=10),
        Slicing(amount=context_length-1, end="left"),
    ]

    for file in files:
        filename: str = file.split(".")[0]

        print(f"Predicting {filename} ({dataset_name}) with {model.__class__.__name__}")

        models_path: str = f"models/{dataset_name}/{filename}/"

        df: pd.DataFrame = pd.read_csv(os.path.join(data_path, file))
        input: np.ndarray = df["input"].to_numpy().flatten()
        output: np.ndarray = df["output"].to_numpy().flatten()

        for t in input_trasnsformations:
            input = t(input)
        for t in output_trasnsformations:
            output = t(output)

        archway = RegressionArchway(
            model=model,
            path=models_path,
            # device="cpu",
        )

        predictions: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        gc.collect()
        for idx, i in enumerate(input):
            before: float = time.time()
            prediction: RegressionPrediction = archway.predict(i)
            after: float = time.time()
            predictions.append(prediction.value)
            stds.append(prediction.std) if prediction.std is not None else None
            Logger.plain(f"-> processing inputs ({int(idx/len(input)*100)}%) | Time per window: {(after-before)*1000:.2f} ms", color="cyan", flush=True)

        predicted_output: np.ndarray = np.array(predictions).flatten()
        predicted_stds: np.ndarray = np.array(stds).flatten() if stds else np.zeros(len(predicted_output))
        true_input: np.ndarray = input[:, -1].flatten()[-len(predicted_output):]
        true_output: np.ndarray = output[-len(predicted_output):]

        info = RegressionEvalInfo(
            true_input=true_input,
            true_output=true_output,
            predicted_output=predicted_output,
        )

        print(info)


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

    inference(strain_model, strain_context_length, "strain")
    inference(stress_model, stress_context_length, "stress")


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

    inference(strain_model, strain_context_length, "strain")
    inference(stress_model, stress_context_length, "stress")


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

    inference(strain_model, strain_context_length, "strain")
    inference(stress_model, stress_context_length, "stress")


def fnn() -> None:
    strain_context_length: int = 720
    strain_model = FeedforwardRegressor(
        input_dim=strain_context_length,
        hidden_dims=[64, 128, 256],
        dropout=0.1,
        use_batch_norm=True,
        use_layer_norm=False,
    )

    stress_context_length: int = 960
    stress_model = FeedforwardRegressor(
        input_dim=stress_context_length,
        hidden_dims=[128, 256, 64],
        dropout=0.1,
        use_batch_norm=True,
        use_layer_norm=False,
    )

    inference(strain_model, strain_context_length, "strain")
    inference(stress_model, stress_context_length, "stress")


if __name__ == "__main__":
    gp()
    transformer()
    mamba()
    fnn()
