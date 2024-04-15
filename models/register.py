


classes={}
def register(cls):
    classes[cls.__name__] = cls
    return cls

def get_class():
    return classes

