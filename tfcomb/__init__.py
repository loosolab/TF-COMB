from importlib import import_module

__version__ = "1.0.2"

#Set classes to be available directly from upper tfcomb, i.e. "from tfcomb import CombObj"
global_classes = ["tfcomb.objects.CombObj",
                  "tfcomb.objects.DiffCombObj",
                  "tfcomb.objects.DistObj"]

for c in global_classes:
    
    module_name = ".".join(c.split(".")[:-1])
    attribute_name = c.split(".")[-1]
    
    module = import_module(module_name)
    attribute = getattr(module, attribute_name)
                           
    globals()[attribute_name] = attribute
