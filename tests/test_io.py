import os
import numpy as np
###############################################################################
from pathlib import Path
###############################################################################
from modelmiezelb.io import NPZExporter, JSONExporter, JSONLoader, ContrastData, IOManager, NPZLoader, Exporter, Loader
###############################################################################
# Path quarrels
testdir = Path(__file__).absolute().parent
resources = testdir / "resources"

def test_IOManager():
    """

    """
    io_npz_json = IOManager(NPZLoader(), JSONExporter())
    io_json_npz = IOManager(JSONLoader(), NPZExporter())

    assert isinstance(io_npz_json, IOManager)
    assert isinstance(io_json_npz.loader, Loader)
    assert isinstance(io_json_npz.exporter, Exporter)

    # Importing a npz file
    cd = io_npz_json.load(resources / "test_NPZExporter.npz")[0]
    print(cd.filename)
    # Exporting it to json format
    io_npz_json.export((resources / "test_JSONExporter.json", cd))

    # Importing the same file again
    cdloaded = io_json_npz.load(resources / "test_JSONExporter.json")[0]
    print(cdloaded.filename)
    # Exporting now to a npz file
    io_json_npz.export((resources / "test_IOManager_npzexport", cdloaded))

#------------------------------------------------------------------------------

def test_NPZloader():
    loader = NPZLoader()
    assert isinstance(loader, Loader)

    for item in loader.load(resources / "test_io_contrastdata_foil0_arc5.npz"):
        print(f"ITEM :\n{item}\n\n")

#------------------------------------------------------------------------------

def test_JSONLoader():
    loader = JSONLoader()
    assert isinstance(loader, Loader)

    loadeddata = loader.load(resources / "T_Tc_minus1_arc8.json")
    for k in loadeddata:
        print(k)

#------------------------------------------------------------------------------

def test_ContrastData():
    """

    """
    loader = NPZLoader()
    cd = ContrastData(*loader.load(resources / "test_io_contrastdata_foil0_arc5.npz"))
    assert isinstance(cd, ContrastData)
    assert len(cd.keys) == len(cd.data)

    print(cd.foilnum)
    print(cd.filename)

#------------------------------------------------------------------------------

def test_NPZExporter():
    """

    """
    # Loading a ContrastData object to get something to export
    loader = NPZLoader()
    cd = ContrastData(*loader.load(resources / "test_io_contrastdata_foil0_arc5.npz"))
    print("Before exort: ", cd.foilnum)
    cd.foilnum = 0
    cd.arcnum = 5
    print("FUCKING fitparams: ", cd.fitparams)

    # Exporting the data again in a NPZ file
    exporter = NPZExporter()
    exportpath = resources / "test_NPZExporter"
    exporter.export(exportpath, cd)

    # Check if the exported data is loadable
    cdloaded = ContrastData(*loader.load(str(exportpath) + ".npz"))
    print(cdloaded.keys)
    print(cdloaded.arcnum)



#------------------------------------------------------------------------------

def test_JSONExporter():
    """

    """
    # Loading a ContrastData object to get something to export
    loader = NPZLoader()
    cd = ContrastData(*loader.load(resources / "test_io_contrastdata_foil0_arc5.npz"))
    cd.foilnum = 0
    cd.arcnum = 5

    # Exporting the data again
    exporter = JSONExporter()
    exportpath = resources / "test_JSONExporter.json"
    exporter.export(exportpath, cd)

    # Check if the exported data is loadable
    jloader = JSONLoader()
    cdloaded = ContrastData(*jloader.load(exportpath))
    print(cdloaded.keys)
    print(cdloaded.arcnum)

#------------------------------------------------------------------------------

if __name__ == "__main__":
    test_IOManager()
#    test_NPZloader()
#    test_JSONLoader()
#    test_ContrastData()
#    test_NPZExporter()
#    test_JSONExporter()