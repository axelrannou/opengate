import wget
import os
import tarfile
import platform
import sys

# Data for Geant4
data_packages = [
    # Geant4 11.0.2
    # "https://cern.ch/geant4-data/datasets/G4NDL.4.6.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4EMLOW.8.0.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4PhotonEvaporation.5.7.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4RadioactiveDecay.5.6.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4PARTICLEXS.4.0.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4PII.1.3.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4RealSurface.2.2.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4SAIDDATA.2.0.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4ABLA.3.1.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4INCL.1.0.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4TENDL.1.4.tar.gz",
    # "https://cern.ch/geant4-data/datasets/G4ENSDFSTATE.2.3.tar.gz",
    #
    # Geant4 11.1.0
    "https://cern.ch/geant4-data/datasets/G4NDL.4.7.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4EMLOW.8.2.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4PhotonEvaporation.5.7.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4RadioactiveDecay.5.6.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4PARTICLEXS.4.0.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4PII.1.3.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4RealSurface.2.2.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4SAIDDATA.2.0.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4ABLA.3.1.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4INCL.1.0.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4TENDL.1.4.tar.gz",
    "https://cern.ch/geant4-data/datasets/G4ENSDFSTATE.2.3.tar.gz",
]


# Check and download Geant4 data if not present:
def check_G4_data_folder():
    dataLocation = get_G4_data_folder()
    if not os.path.exists(dataLocation):
        print("No Geant4 data available in: " + dataLocation)
        print("I download it for you.")
        download_G4_data()
        print("")
        print("Done")
        return True


def check_G4_data():
    # check if the G4 data folder is there
    folder_was_missing = check_G4_data_folder()
    if folder_was_missing is True:
        return

    # Check if the G4 data folder is up to date
    missing_data = []
    g4_data_paths = get_G4_data_path()
    for p in g4_data_paths.values():
        if not os.path.exists(p):
            missing_data.append(p)
    if len(missing_data) > 0:
        print("Some Geant4 data is missing, namely:")
        for p in missing_data:
            print(p)
        print("I download a fresh G4 dataset for you.")
        # download_G4_data()
        print("download_G4_data")
        print("")
        print("Done")


# Download Geant4 data:
def download_G4_data():
    dataLocation = get_G4_data_folder()
    if not os.path.exists(dataLocation):
        os.mkdir(dataLocation)
    base_names = set()
    for i, package in enumerate(data_packages):
        print(f"\nDownloading {i+1}/{len(data_packages)} {package}")
        packageArchive = wget.download(package, out=dataLocation)
        with tarfile.open(packageArchive) as tar:
            tar.extractall(path=dataLocation)
            # extract the base folders from the tar archive
            # into which the G4 data was extracted
            base_names.update(
                [
                    os.path.split(f)[0]
                    for f in tar.getnames()
                    if os.path.split(f)[0] != ""
                ]
            )
        os.remove(packageArchive)

    # Check if there are old data and avert the user
    outdated_paths = []
    for f in os.listdir(dataLocation):
        p = os.path.join(dataLocation, f)
        if os.path.isdir(p) and p not in base_names:
            outdated_paths.append(p)
    print("")
    print("The following folders are outdated and can be deleted safely:")
    for f in outdated_paths:
        print(f)


# Return Geant4 data folder:
def get_G4_data_folder():
    packageLocation = os.path.dirname(os.path.realpath(__file__))
    dataLocation = os.path.join(packageLocation, "geant4_data")
    return dataLocation


# Return Geant4 data path:
def get_G4_data_path():
    dataLocation = get_G4_data_folder()
    # 10.6
    # g4DataPath = {
    #     "G4NEUTRONHPDATA": os.path.join(dataLocation, 'G4NDL4.6'),
    #     "G4LEDATA": os.path.join(dataLocation, 'G4EMLOW7.9.1'),
    #     "G4LEVELGAMMADATA": os.path.join(dataLocation, 'PhotonEvaporation5.5'),
    #     "G4RADIOACTIVEDATA": os.path.join(dataLocation, 'G4RadioactiveDecay5.4'),
    #     "G4SAIDXSDATA": os.path.join(dataLocation, 'G4SAIDDATA2.0'),
    #     "G4PARTICLEXSDATA": os.path.join(dataLocation, 'G4PARTICLEXS2.1'),
    #     "G4ABLADATA": os.path.join(dataLocation, 'G4ABLA3.1'),
    #     "G4INCLDATA": os.path.join(dataLocation, 'G4INCL1.0'),
    #     "G4PIIDATA": os.path.join(dataLocation, 'G4PII1.3'),
    #     "G4ENSDFSTATEDATA": os.path.join(dataLocation, 'G4ENSDFSTATE2.2'),
    #     "G4REALSURFACEDATA": os.path.join(dataLocation, 'G4RealSurface2.1.1')
    # }
    # 10.7
    # g4DataPath = {
    #     "G4NEUTRONHPDATA": os.path.join(dataLocation, 'G4NDL4.6'),
    #     "G4LEDATA": os.path.join(dataLocation, 'G4EMLOW7.13'),
    #     "G4LEVELGAMMADATA": os.path.join(dataLocation, 'PhotonEvaporation5.7'),
    #     "G4RADIOACTIVEDATA": os.path.join(dataLocation, 'RadioactiveDecay5.6'),
    #     "G4SAIDXSDATA": os.path.join(dataLocation, 'G4SAIDDATA2.0'),
    #     "G4PARTICLEXSDATA": os.path.join(dataLocation, 'G4PARTICLEXS3.1.1'),  # to update ? how ?
    #     "G4ABLADATA": os.path.join(dataLocation, 'G4ABLA3.1'),
    #     "G4INCLDATA": os.path.join(dataLocation, 'G4INCL1.0'),
    #     "G4PIIDATA": os.path.join(dataLocation, 'G4PII1.3'),
    #     "G4ENSDFSTATEDATA": os.path.join(dataLocation, 'G4ENSDFSTATE2.3'),
    #     "G4REALSURFACEDATA": os.path.join(dataLocation, 'G4RealSurface2.2')
    # }
    # 11
    # g4DataPath = {
    #     "G4NEUTRONHPDATA": os.path.join(dataLocation, "G4NDL4.6"),
    #     "G4LEDATA": os.path.join(dataLocation, "G4EMLOW8.0"),
    #     "G4LEVELGAMMADATA": os.path.join(dataLocation, "PhotonEvaporation5.7"),
    #     "G4RADIOACTIVEDATA": os.path.join(dataLocation, "RadioactiveDecay5.6"),
    #     "G4SAIDXSDATA": os.path.join(dataLocation, "G4SAIDDATA2.0"),
    #     "G4PARTICLEXSDATA": os.path.join(dataLocation, "G4PARTICLEXS4.0"),
    #     "G4ABLADATA": os.path.join(dataLocation, "G4ABLA3.1"),
    #     "G4INCLDATA": os.path.join(dataLocation, "G4INCL1.0"),
    #     "G4PIIDATA": os.path.join(dataLocation, "G4PII1.3"),
    #     "G4ENSDFSTATEDATA": os.path.join(dataLocation, "G4ENSDFSTATE2.3"),
    #     "G4REALSURFACEDATA": os.path.join(dataLoc   ation, "G4RealSurface2.2"),
    # }
    # 11.1
    g4DataPath = {
        "G4NEUTRONHPDATA": os.path.join(dataLocation, "G4NDL4.7"),
        "G4LEDATA": os.path.join(dataLocation, "G4EMLOW8.2"),
        "G4LEVELGAMMADATA": os.path.join(dataLocation, "PhotonEvaporation5.7"),
        "G4RADIOACTIVEDATA": os.path.join(dataLocation, "RadioactiveDecay5.6"),
        "G4SAIDXSDATA": os.path.join(dataLocation, "G4SAIDDATA2.0"),
        "G4PARTICLEXSDATA": os.path.join(dataLocation, "G4PARTICLEXS4.0"),
        "G4ABLADATA": os.path.join(dataLocation, "G4ABLA3.1"),
        "G4INCLDATA": os.path.join(dataLocation, "G4INCL1.0"),
        "G4PIIDATA": os.path.join(dataLocation, "G4PII1.3"),
        "G4ENSDFSTATEDATA": os.path.join(dataLocation, "G4ENSDFSTATE2.3"),
        "G4REALSURFACEDATA": os.path.join(dataLocation, "G4RealSurface2.2"),
    }
    return g4DataPath


# Set Geant4 data paths:
def set_G4_data_path():
    g4DataPath = get_G4_data_path()
    for key, value in g4DataPath.items():
        os.environ[key] = value
    s = platform.system()
    if s == "Linux":
        g4libFolder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../opengate_core.libs"
        )
    elif s == "Darwin":
        g4libFolder = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), ".dylibs"
        )
    # print('DEBUG: current Geant4 lib', g4libFolder)
    # print('DEBUG: current Geant4 data', get_G4_data_folder())
    if s == "Windows":
        os.add_dll_directory(g4libFolder)
    else:
        sys.path.append(g4libFolder)
    # sys.path.append(opengate_core_folder)
    if not "LD_LIBRARY_PATH" in os.environ:
        os.environ["LD_LIBRARY_PATH"] = ""
    os.environ["LD_LIBRARY_PATH"] = g4libFolder + ":" + os.environ["LD_LIBRARY_PATH"]
