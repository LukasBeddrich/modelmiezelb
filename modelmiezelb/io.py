"""

"""

import json
from pathlib import Path
from numpy import array, load, savez_compressed

class IOManager:
    """
    Structure for exporting and importing MIEZE fit results and data with JSON
    """
    def __init__(self, io_path, name, loader=None, exporter=None):
        """
        Constructs a Exporter instance.
        If only the io_path is given to constructor, it tries to load data from this file.

        Parameters
        ----------
        io_path         :   str
            path from which data will be loader or stored
        name            :   str
            a name
        loader          :   load_structure.Loader, None
            Loader object written for the 'iron' and 'nickel' data
            If not provided 'data' and 'meta_data' need to be given
        exporter        :   .Exporter, None
            Exporter obj
        """
        self.io_path = io_path
        self.name = name
        self.loader = loader
        self.exporter = exporter

#------------------------------------------------------------------------------

    def populate(self, data=None, meta_data=None, fmin=None, params=None, minuit=None):
        """
        data            :   dict, None
            dict object as given by Loader.all_data
            If not given, a valid load_structure.Loader is required
        meta_data       :   dict, None
            dict object as given by Loader.meta_data
            If not given, a valid load_structure.Loader is required
        fmin            :    
        params          :   iminuit.utils.Params, dict
            Parameters of th efit result
        minuit   :   iminuit.utils.Params, dict
            from this instance, the fit values of the parameters are extracted
        """
        self.data = data
        self.meta_data = meta_data
        self.meta_data["name"] = self.name
        self.fmin = fmin
        self.params = params
        try:
            self.fmin = minuit.fmin
            self.params = minuit.params
        except:
            print("Retrieving information from Minuit obj failed.")

#------------------------------------------------------------------------------

    def load(self):
        """

        """
        try:
            self.populate(**self.loader.load(self.io_path))
        except:
            print("Loading failed.")

#------------------------------------------------------------------------------

    def export(self):
        """

        """
        # This could be replaced in the future
        export_data = dict(
             params={p.name : (p.value, p.error) for p in self.params},
            #  fit_params={p.name : (p.value, p.error) for p in self.params},
             fmin=dict(
                 valid=self.fmin.is_valid,
                 edm=self.fmin.edm,
                 chi2=self.fmin.fval
             ),
             data=self.data,
             meta_data=self.meta_data
        )
        try:
            self.exporter.export(self.io_path, **export_data)
        except:
            print("Export failed.")

###############################################################################
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
            where each parameter has a name and a list [value, error]
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

    """
    def load(self, io_path):
        """

        """
        pass

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
                    foilnum = npzfile[k]
                elif k == "arcnum":
                    arcnum = npzfile[k]
                elif k == "fitparams":
                    fitparams = npzfile[k][()]
                elif k == "description":
                    descrstr = npzfile[k]
                else:
                    keys.append(k)
                    data.append(npzfile[k])

        data = array(data)
        
        return filename, keys, data, foilnum, arcnum, fitparams, descrstr

        # foilnum, arcnum = self.parse_filename(io_path)
        # keys, data = self._extract_from_npz

        # if self.rootpath:
        #     filename = os.path.join(self.rootpath, filename)
        #     with np.load(filename) as npzfile:
        #         keys, data = self._extract_from_npz(npzfile)

        # else:
        #     with np.load(filename) as npzfile:
        #         keys, data = self._extract_from_npz(npzfile)

###############################################################################
###############################################################################
###############################################################################

class Exporter:
    """

    """
    def export(self, io_path, **export_data):
        """

        """
        pass

#------------------------------------------------------------------------------

    def _collect_export_data(self, contrastdata):
        """

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

class LegacyJSONExporter(Exporter):
    """

    """
    def export(self, io_path, **export_data):
        """

        """
        with open(io_path, "w") as export_file:
            json.dump(export_data, export_file)

###############################################################################

class JSONExporter(Exporter):
    """

    """
    def export(self, io_path, contrastdata):
        """

        """
        # assert io_path == contrastdata.filename
        export_data = self._collect_export_data(contrastdata)
        export_data.update({"filename" : str(io_path)})

        with open(io_path, "w") as export_file:
            json.dump(export_data, export_file, indent=4)

#------------------------------------------------------------------------------

    # def _collect_export_data(self, contrastdata):
    #     """

    #     """
    #     export_data = dict(
    #         foilnum     = contrastdata.foilnum,
    #         arcnum      = contrastdata.arcnum,
    #         fitparams   = contrastdata.fitparams,
    #         description = contrastdata.description
    #     )
    #     export_data.update(
    #         dict(
    #             zip(
    #                 contrastdata.keys,
    #                 contrastdata.data.tolist()
    #             )
    #         )
    #     )
    #     return export_data

###############################################################################

class NPZExporter(Exporter):
    """

    """
    def export(self, io_path, contrastdata):
        """

        """

        # assert io_path == contrastdata.filename
        export_data = self._collect_export_data(contrastdata)
        # export_data = dict(
        #     filename    = contrastdata.filename,
        #     foilnum     = contrastdata.foilnum,
        #     arcnum      = contrastdata.arcnum,
        #     fitparams   = contrastdata.fitparams,
        #     description = contrastdata.descrstr
        # )
        # export_data.update(dict(zip(contrastdata.keys, contrastdata.data)))

        savez_compressed(io_path, **export_data)

###############################################################################