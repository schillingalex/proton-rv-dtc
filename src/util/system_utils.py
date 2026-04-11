import shutil


def is_software_available(command: str) -> bool:
    """
    Is the given command available on the system?
    Example: is_software_available("singularity") can be used to determine which containerization to use.

    :param command: Program name to check
    :return: True if the program is available, False otherwise
    """
    return shutil.which(command) is not None
