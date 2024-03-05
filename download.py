import gdown

import os

from vclog import Logger


def create_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)


def delete_files(directory: str) -> None:
    for filename in os.listdir(directory):
        os.remove(directory + filename)


def create_or_delete_files(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    else:
        for filename in os.listdir(directory):
            os.remove(directory + filename)


def directory_checks(stress_data_dir: str, strain_data_dir: str) -> None:
    if not os.path.exists(stress_data_dir) or not os.path.exists(strain_data_dir):
        raise Exception("Stress or strain data directory does not exist")

    if not len(os.listdir(stress_data_dir)):
        raise Exception("No stress files found")

    if not len(os.listdir(strain_data_dir)):
        raise Exception("No strain files found")


def structure() -> None:
    stress_data_dir: str = "data/nylon_elastic_wire/stress/"
    strain_data_dir: str = "data/nylon_elastic_wire/strain/"
    paper_data_dir: str = "data/nylon_elastic_wire/paper/"
    paper_strain_data_dir: str = os.path.join(paper_data_dir, "strain/")
    paper_stress_data_dir: str = os.path.join(paper_data_dir, "stress/")

    directory_checks(stress_data_dir, strain_data_dir)
    create_directory(paper_data_dir)
    create_directory(paper_stress_data_dir)
    create_directory(paper_strain_data_dir)

    stress_filenames: list[str] = os.listdir(stress_data_dir)
    stress_filenames.sort()

    strain_filenames: list[str] = os.listdir(strain_data_dir)
    strain_filenames.sort()

    Logger.info("STRESS")
    for filename in stress_filenames:
        name: str = filename.split(".")[0]
        dir_name: str = paper_stress_data_dir + name + "/"
        create_or_delete_files(dir_name)

        Logger.info(f"Processing {name}")
        os.system(f"cp {stress_data_dir}* {dir_name}")
        os.system(f"mv {dir_name}{filename} {dir_name}zz{filename}")

    print()
    Logger.info("STRAIN")
    for filename in strain_filenames:
        name: str = filename.split(".")[0]

        dir_name: str = paper_strain_data_dir + name + "/"
        create_or_delete_files(dir_name)

        Logger.info(f"Processing {name}")
        os.system(f"cp {strain_data_dir}* {dir_name}")
        os.system(f"mv {dir_name}{filename} {dir_name}zz{filename}")


def download() -> None:
    nylon_elastic_wire_url: str = "https://drive.google.com/drive/folders/1DQnxLZCCgcCv9BXFKNBR8RIi0E_foTqF?usp=sharing"
    save_path: str = "data/"

    gdown.download_folder(nylon_elastic_wire_url, output=save_path, quiet=False)


if __name__ == "__main__":
    download()
    structure()
