import os


def ListFiles(Directory, Extension):
    '''
    Given an extension, get the file names for a Directory in a sorted order. Rerurn an empty list if Directory == "".

    :param String Directory: absolute path of a file directory
    :param String Extension: Extension of the files you want. Better include "." in the Extension
    :return ListOfFiles: The list of absolute paths of the files you want under Directory
    :rtype List[String]
    '''
    ListOfFiles = []
    if (Directory == "" or Directory == []):
        return []
    if (type(Directory) != list and os.path.isdir(Directory) == False):
        raise ValueError(Directory, 'Directory is not a directory!')
    if (type(Extension) != str):
        raise ValueError(Extension, 'Extension is not a string!')
    if (Extension):
        if (Extension[0] != "."):
            Extension = "." + Extension[0]
    if type(Directory) == list:
        Directories = Directory
        for Directory in Directories:
            filenames = os.listdir(Directory)
            for filename in filenames:
                # list filenames
                # get the absolute path for the files
                AbsolutePath = os.path.abspath(
                    os.path.join(Directory, filename))
                # get the absolute path for the files
                if os.path.splitext(filename)[1] == Extension:
                    if os.path.isfile(AbsolutePath):
                        ListOfFiles.append(AbsolutePath)
    else:
        filenames = os.listdir(Directory)
        for filename in filenames:
            # list filenames
            # get the absolute path for the files
            AbsolutePath = os.path.abspath(os.path.join(Directory, filename))
            # get the absolute path for the files
            if os.path.splitext(filename)[1] == Extension:
                if os.path.isfile(AbsolutePath):
                    ListOfFiles.append(AbsolutePath)
    return sorted(ListOfFiles)
