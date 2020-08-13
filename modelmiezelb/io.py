"""

"""

import json

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

        """
        with open(io_path, "r") as load_file:
            loaded_data = json.load(load_file)
        return loaded_data

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

###############################################################################

class JSONExporter(Exporter):
    """

    """
    def export(self, io_path, **export_data):
        """

        """
        with open(io_path, "w") as export_file:
            json.dump(export_data, export_file)