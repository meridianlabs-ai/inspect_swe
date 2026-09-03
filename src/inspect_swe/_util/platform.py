from typing import no_type_check


@no_type_check
def running_in_notebook() -> bool:
    try:
        from IPython.core.getipython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True
