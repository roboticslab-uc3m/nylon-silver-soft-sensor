import os
import time
import gc

import numpy as np
import pandas as pd

from catasta import Archway
from catasta.models import (
    ApproximateGPRegressor,
    TransformerRegressor,
    MambaRegressor,
    FeedforwardRegressor,
)
from catasta.dataclasses import PredictionInfo, RegressionEvalInfo
from catasta.transformations import (
    Normalization,
    Decimation,
    WindowSliding,
    Slicing,
    Custom,
)

from vclog import Logger


def inference(model_name: str, context_length: int, dataset_name: str) -> None:
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

        print(f"Predicting {filename} ({dataset_name}) with {model_name}")

        models_path: str = f"models/{dataset_name}/{filename}/{model_name}/"

        df: pd.DataFrame = pd.read_csv(os.path.join(data_path, file))
        input: np.ndarray = df["input"].to_numpy().flatten()
        output: np.ndarray = df["output"].to_numpy().flatten()

        for t in input_trasnsformations:
            input = t(input)
        for t in output_trasnsformations:
            output = t(output)

        archway = Archway(
            path=models_path,
            # device="cpu",
        )

        predictions: list[np.ndarray] = []
        stds: list[np.ndarray] = []
        times: list[float] = []
        gc.collect()
        for idx, i in enumerate(input):
            before: float = time.time()
            prediction: PredictionInfo = archway.predict(i.reshape(1, -1))
            after: float = time.time()
            times.append((after-before) * 1000)
            predictions.append(prediction.value)
            stds.append(prediction.std) if prediction.std is not None else None
            Logger.plain(f"-> processing inputs ({int(idx/len(input)*100)}%) | Time per window: {(after-before)*1000:.2f} ms", color="cyan", flush=True)

        Logger.plain(f"-> processing inputs (100%) | Time per window: {np.mean(times):.2f} ms", color="cyan")

        predicted_output: np.ndarray = np.array(predictions).flatten()
        predicted_stds: np.ndarray = np.array(stds).flatten() if stds else np.zeros(len(predicted_output))
        true_input: np.ndarray = input[:, -1].flatten()[-len(predicted_output):]
        true_output: np.ndarray = output[-len(predicted_output):]

        info = RegressionEvalInfo(
            true_input=true_input,
            true_output=true_output,
            predicted_output=predicted_output,
            predicted_std=predicted_stds,
        )

        print(info)


def gp() -> None:
    model_name: str = "ApproximateGPRegressor"
    strain_context_length: int = 1536
    stress_context_length: int = 720

    inference(model_name, strain_context_length, "strain")
    inference(model_name, stress_context_length, "stress")


def transformer() -> None:
    model_name: str = "TransformerRegressor"
    strain_context_length: int = 1920
    stress_context_length: int = 720

    inference(model_name, strain_context_length, "strain")
    inference(model_name, stress_context_length, "stress")


def mamba() -> None:
    model_name: str = "MambaRegressor"
    strain_context_length: int = 1152
    stress_context_length: int = 1280

    inference(model_name, strain_context_length, "strain")
    inference(model_name, stress_context_length, "stress")


def fnn() -> None:
    model_name: str = "FeedforwardRegressor"
    strain_context_length: int = 720
    stress_context_length: int = 960

    inference(model_name, strain_context_length, "strain")
    inference(model_name, stress_context_length, "stress")


if __name__ == "__main__":
    gp()
    transformer()
    mamba()
    fnn()
