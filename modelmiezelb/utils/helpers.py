from copy import deepcopy
from collections import deque


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    q = deque()
    for current in nested_list:
        if isinstance(current, list):
            q.extendleft(reversed(current))
        else:
            q.appendleft(current)
        while q:
            toFlat = q.popleft()
            if isinstance(toFlat, list):
                q.extendleft(reversed(toFlat))
            else:
                yield toFlat

#------------------------------------------------------------------------------

def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

#------------------------------------------------------------------------------

def get_key_for_grouping(item):
    """
    Returns the key for grouping params during update of SqE.

    Parameters
    ----------
    item    : tuple
        (key, value) as given by dict.items()
    """
    try:
        return item[0].split("_")[1] # This should be the name of the line
    except IndexError:
        return "model_params"

#------------------------------------------------------------------------------

def format_param_dict_for_logger(params_dict):
    """

    """
    maxchars = max([len(key) for key in params_dict.keys()])
    return "\n".join([f"{key.rjust(maxchars)} : {val}" for key, val in params_dict.items()])

#------------------------------------------------------------------------------

def format_sqt_lines_for_logger(sqt):
    """

    """
    maxchars = max([len(line.name) for line in sqt.sqemodel._lines])

    return "\n\n".join([f"{line.name.rjust(maxchars)} :\n{'-' * (maxchars + 2)}\n{format_param_dict_for_logger(line.line_params)}" for line in sqt.sqemodel._lines])

#------------------------------------------------------------------------------

def minuit_to_dict(minuit):
    """

    """
    fitparams = {"params" : {}}
    fmin = minuit.fmin
    params = minuit.params

    fitparams["fval"] = fmin.fval
    fitparams["edm"] = fmin.edm
    fitparams["is_valid"] = fmin.is_valid
    fitparams["ncalls"] = fmin.ncalls

    for param in params:
        fitparams["params"][param["name"]] = (param["value"], param["error"], param["is_fixed"])

    return fitparams

#------------------------------------------------------------------------------

def results_to_dict(params, fmin):
    """

    """
    fitparams = {"params" : {}}

    fitparams["fval"] = fmin.fval
    fitparams["edm"] = fmin.edm
    fitparams["is_valid"] = fmin.is_valid
    fitparams["ncalls"] = fmin.ncalls

    for param in params:
        fitparams["params"][param["name"]] = (param["value"], param["error"], param["is_fixed"])

    return fitparams

#------------------------------------------------------------------------------