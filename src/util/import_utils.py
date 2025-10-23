

def import_from_string(import_str):
    """
    Source: https://stackoverflow.com/a/8255024

    Imports an entity given its fully qualified name, e.g. cluster.diffusion.CauchyKernelDiffuser will import the class
    CauchyKernelDiffuser from the package cluster and the file diffusion.py.
    """
    last_separator = import_str.rfind(".")
    classname = import_str[last_separator+1:len(import_str)]
    module = __import__(import_str[0:last_separator], globals(), locals(), [classname])
    return getattr(module, classname)


def instance_from_string(class_name, *args, **kwargs):
    """
    Imports the class by its fully qualified name, e.g., cluster.diffusion.CauchyKernelDiffuser,
    and instantiates it with the parameters given as positional args and keyword args.
    """
    imported_class = import_from_string(class_name)
    return imported_class(*args, **kwargs)
