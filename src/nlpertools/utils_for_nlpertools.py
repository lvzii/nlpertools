from importlib import import_module


def try_import(name, package):
    try:
        return import_module(name, package=package)
    except:
        print("import {} failed".format(name))
    finally:
        pass
