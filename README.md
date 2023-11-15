# Design of a Soft Sensor based on Silver-Coated Polyamide Threads and Stress-Strain Modeling via Gaussian Processes

This is the official implementation of the paper **Design of a Soft Sensor based on Silver-Coated Polyamide Threads and Stress-Strain Modeling via Gaussian Processes**.

## Abstract

The demand for reliable and efficient soft sensors has grown exponentially with the evolution of wearable devices and smart textiles. However, existing soft sensors often face challenges related to hysteresis, noise, and accuracy, hindering their seamless integration into practical applications. Our research presents a pioneering stress-strain soft sensor model based on silver-coated polyamide threads, augmented with additional silicone and graphite coatings for enhanced properties. Specifically, the silicone coating proves instrumental in elevating the sensor’s gauge factor and reducing noise, resulting in heightened accuracy. The extensive data analysis reveals the presence of hysteresis and non-linearities; however, our data exhibits remarkable robustness, as indicated by the high Spearman correlation coefficient values. In the context of system identification, a comparative analysis between traditional regression methods and Gaussian Process Regression (GPs Regression) demonstrates the superior performance of GPs: this technique outperforms conventional regression techniques, obtaining 8.75±4.06% Root Mean Square Error (RMSE) compared to the 12.70±7.04% error observed in traditional methods. This research not only advances the field of soft sensor technology by developing an accurate, affordable and adaptable device, but also offers valuable insights into highly effective system identification techniques tailored for wearable devices.

## Contact

Corresponding authors:
- cballest@pa.uc3m.es
- vimunozs@pa.uc3m.es

## Installation

To have an isolated testing, we recommend installing a virtual environment like [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/). If you don't know how to install it, check out [this tutorial](https://vistormu.github.io/posts/conda.html).

Once you have your virtual environment activated, clone the repository:

```bash
git clone https://github.com/roboticslab-uc3m/nylon-silver-soft-sensor.git
```

and install the dependencies:

```bash
pip install -r requirements
```

Then, run the `main.py` file:

```bash
python main.py
```

## Citation

(Under review)
