# Silver-Coated Polyamide Threads Stress-Strain Soft Sensor Modeling

This is the official repository of the implementation of the modeling of the silver-coated sof strain-stress sensor. The implementation is made using [catasta](https://github.com/vistormu/catasta), a simple custom library that facilitates the training and inference of Deep Learning models.

The code applies for the following papers:

- Design of a Soft Sensor based on Silver-Coated Polyamide Threads and Stress-Strain Modeling via Gaussian Processes
- Flexible strain-stress soft sensor modeling: integration and comparative analysis of Deep Learning architectures for regression


## Installation

We recommend installing a virtual environment like [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/), and install a version of Python higher that 3.10.x, as the code uses type hinting.

Once you have your virtual environment activated, clone the repository:

```bash
git clone https://github.com/roboticslab-uc3m/nylon-silver-soft-sensor.git
```

and install the dependencies:

```bash
pip install -r requirements
```

## Reproducibility

To reproduce the paper results, first, download the data using:

```python
python download.py
```

If it fails, download the data [for here](https://drive.google.com/drive/folders/1DQnxLZCCgcCv9BXFKNBR8RIi0E_foTqF?usp=sharing), place the dataset in `data/`, comment the `download()` function in `download.py` and run again the script.

To train all models, run:

```bash
python train.py
```

To make the predictions using the pre-trained models, use:

```bash
python inference.py
```


## Contact

Corresponding authors:
- vimunozs@pa.uc3m.es
- cballest@pa.uc3m.es

## Citation

#### Design of a Soft Sensor based on Silver-Coated Polyamide Threads and Stress-Strain Modeling via Gaussian Processes

```bib
@article{Ballester_2024,
 author = {Ballester, Carmen and Muñoz, Víctor and Copaci, Dorin and Moreno, Luis and Blanco, Dolores},
 doi = {10.1016/j.sna.2024.115058},
 issn = {0924-4247},
 journal = {Sensors and Actuators A: Physical},
 month = {March},
 pages = {115058},
 publisher = {Elsevier BV},
 title = {Design of a soft sensor based on silver-coated polyamide threads and stress-strain modeling via Gaussian processes},
 url = {http://dx.doi.org/10.1016/j.sna.2024.115058},
 volume = {367},
 year = {2024}
}
```

#### Flexible strain-stress soft sensor modeling: integration and comparative analysis of Deep Learning architectures for regression

(under review)
