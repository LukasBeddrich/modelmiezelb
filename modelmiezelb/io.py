"""

"""

import json
from pathlib import Path
from numpy import array, load, savez_compressed

class IOManager:
    """
    Structure for exporting and importing MIEZE fit results and data to files
    """
    def __init__(self, loader=None, exporter=None):
        """
        Constructs a IOManager instance.

        Parameters
        ----------
        loader          :   None, modelmiezelb.io.Loader or subclass thereof
            Loader instance which implements loading of a particular file format
        exporter        :   None, modelmiezelb.io.Exporter of subclass thereof
            Exporter obj which implements exporting to a particular file format
        """
        self.loader = loader
        self.exporter = exporter

#------------------------------------------------------------------------------

    def load(self, *io_path):
        """
        Uses the stored loader to generate ContrastData instances with the data
        from the files

        Parameters
        ----------
        io_path :   any number of str, pathlib.Path objects
            pointing to a file for loading the data

        Return
        ------
        data    :   list
            list of ContrastData objects with the loaded data of None if loading
            failed

        Note
        ----
        Will print out the index and filename of the ones unable to load
        """
        data = [None] * len(io_path)
        for idx, path in enumerate(io_path):
            try:
                data[idx] = ContrastData(*self.loader.load(path))
            except:
                print(f"Loading failed for file indexed {idx} named: {path}")
        return data

#------------------------------------------------------------------------------

    def export(self, *path_data_tuples, use_stored_name=False):
        """
        Uses the stored loader to generate ContrastData instances with the data
        from the files

        Parameters
        ----------
        path_data_tuples :   any number of tuples
            expects input of the form (io_path, ContrastData object)

        Note
        ----
        Will print out the index and filename of the ones unable to export
        """
        for idx, (path, cdobj) in enumerate(path_data_tuples):
            try:
                self.exporter.export(path, cdobj)
            except:
                print(f"Exporting failed for file indexed {idx} named: {path}")

###############################################################################
###############################################################################
###############################################################################

class ContrastData:
    """

    """
    def __init__(self, filename, keys, data, foilnum, arcnum, fitparams, descrstr=''):
        """
        Contains contrast data of 1 foil and 1 (q-)mask, but all keys (of one miezepy-environment)

        Basic object to store contrast data after reduction of mieze rawdata

        Parameters
        ----------

        filename  :   str
            specifies its the name of the file it was imported from
        keys      :   list
            list of str, values specifying a data set in self.data        
        data      :   ndarray, empty list
            data set for each key -> len(keys) equals len(data)
        foilnum   :   int, NoneType
            integer value of the foil. not comparable between experiments
        arcnum    :   int, NoneType
            integer describing a mask in contrast postprocessing
        fitparams :   dict, NoneType
            parameters and values associated with a model fit to this data
            All optimized model parameters contained in self.fitparams["params"]
            where each parameter has a name and a tuple (value, error, is_fixed)
            everthing else is optional.
        descrstr  :   str
            A further description of the data set
        """
        self.filename    = filename
        self.keys        = keys
        self.data        = data
        self.foilnum     = foilnum
        self.arcnum      = arcnum
        self.fitparams   = fitparams
        self.description = descrstr

#------------------------------------------------------------------------------

    def get(self, specifier):
        """
        
        """
        assert isinstance(specifier, str)

        spec = specifier.lower()
        if spec == 'filename':
            return self.filename
        elif spec == 'keys':
            return self.keys
        elif spec == 'data':
            return self.data
        elif spec == 'foilnum':
            return self.foilnum
        elif spec == 'arcnum':
            return self.arcnum
        elif spec == 'fitparams':
            return self.fitparams
        elif spec == 'description':
            return self.description
        else:
            print(f"{spec} Not a valid specifier!")
            return None

#------------------------------------------------------------------------------

    def getData(self, index=None, key=None):
        """
        
        """
        try:
            if key:
                index = self.keys.index(key)
            else:
                pass
        except ValueError:
            raise KeyError(f"{key} is not a valid key to retrieve data. Call ContrastData.get('keys') to show available keys")

        return self.data[index]

###############################################################################
###############################################################################
###############################################################################

class Loader:
    """
    Interface for importer classes
    """
    def load(self, io_path):
        """
        Needs to be implmement by any Loader subclass.
        Is the implementation for loading a particular file format

        Parameters
        ----------
        io_path       :   str, pathlib.Path
            specifies file for data retrieval
        """
        raise NotImplementedError("This is the template for the Loader \
            interface.\nDoes not implement any loading strategy.")

###############################################################################

class JSONLoader(Loader):
    """

    """
    def load(self, io_path):
        """
        Grabs all info from npz-file directly and compares with keys with
        standard specifiers. Everything else will be plugged into keys, data

        Parameters
        ----------
        io_path :   str, pathlib.Path
            specifies json-file for data retrieval

        Return
        ------
        filename, keys, data, foilnum, arcnum, fitparams, descrstr
        """
        with open(io_path, "r") as load_file:
            loaded_data = json.load(load_file)
        
        filename  = str(io_path)
        keys      = []
        data      = []
        foilnum   = None
        arcnum    = None
        fitparams = {}
        descrstr  = ''

    
        for k in loaded_data:
            if k == "filename":
                assert filename == loaded_data[k]
            elif k == "foilnum":
                foilnum = loaded_data[k]
            elif k == "arcnum":
                arcnum = loaded_data[k]
            elif k == "fitparams":
                fitparams = loaded_data[k]
            elif k == "description":
                descrstr = loaded_data[k]
            else:
                keys.append(k)
                data.append(loaded_data[k])

        data = array(data)

        return filename, keys, data, foilnum, arcnum, fitparams, descrstr

###############################################################################

class NPZLoader(Loader):
    """

    """
    def load(self, io_path):
        """
        Grabs all info from npz-file directly and compares with keys with
        standard specifiers. Everything else will be plugged into keys, data

        Parameters
        ----------
        io_path :   str, pathlib.Path
            specifies npz-file for data retrieval

        Return
        ------
        filename, keys, data, foilnum, arcnum, fitparams, descrstr

        Note
        ----
        Vgl.: ~/Documents/Nickel/MIEZE/miezepy_analysis/Individualfoils.ipynb
        """

        filename  = str(io_path)
        keys      = []
        data      = []
        foilnum   = None
        arcnum    = None
        fitparams = {}
        descrstr  = ''

        with load(io_path, allow_pickle=True) as npzfile:
            for k in npzfile.files:
                if k == "filename":
                    pass
                elif k == "foilnum":
                    foilnum = npzfile[k].tolist()
                elif k == "arcnum":
                    arcnum = npzfile[k].tolist()
                elif k == "fitparams":
                    fitparams = npzfile[k].tolist()
                elif k == "description":
                    descrstr = npzfile[k].tolist()
                else:
                    keys.append(k)
                    data.append(npzfile[k])

        data = array(data)
        
        return filename, keys, data, foilnum, arcnum, fitparams, descrstr

###############################################################################
###############################################################################
###############################################################################

class Exporter:
    """
    Interface for exporter classes 
    """
    def export(self, io_path, contrastdata):
        """
        Needs to be implmement by any Exporter subclass.
        Is the implementation of a particular file format exportation

        Parameters
        ----------
        io_path       :   str, pathlib.Path
            specifies file for data export
        contrastdata  :   modelmiezelb.io.ContrastData
            a instance storing data for exporting
        """
        raise NotImplementedError("This is the template for the Exporter \
            interface.\nDoes not implement any export strategy.")

#------------------------------------------------------------------------------

    def _collect_export_data(self, contrastdata):
        """
        Collects and structures data from a ContrastData object
        in a dictionary for export
        """
        export_data = dict(
            foilnum     = contrastdata.foilnum,
            arcnum      = contrastdata.arcnum,
            fitparams   = contrastdata.fitparams,
            description = contrastdata.description
        )
        export_data.update(dict(zip(contrastdata.keys, contrastdata.data.tolist())))
        return export_data

###############################################################################

class JSONExporter(Exporter):
    """

    """
    def export(self, io_path, contrastdata):
        """
        Exports to json format

        Parameters
        ----------
        io_path :   str, pathlib.Path
            specifies json-file for data export
        contrastdata :   ContrastData
            see modelmiezelb.io.ContrastData class for documentation
        """
        # assert io_path == contrastdata.filename
        export_data = self._collect_export_data(contrastdata)
        export_data.update({"filename" : str(io_path)})

        with open(io_path, "w") as export_file:
            json.dump(export_data, export_file, indent=4)

###############################################################################

class NPZExporter(Exporter):
    """

    """
    def export(self, io_path, contrastdata):
        """
        Exports to npz format

        Parameters
        ----------
        io_path :   str, pathlib.Path
            specifies npz-file for data export
        contrastdata :   ContrastData
            see modelmiezelb.io.ContrastData class for documentation
        """

        export_data = self._collect_export_data(contrastdata)

        savez_compressed(io_path, **export_data)

###############################################################################