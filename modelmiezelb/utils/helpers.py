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