from vegamdb import VegamDB

db = VegamDB()

print("vegamdb works! Version:", __import__("vegamdb").__version__)

from vegamdb import IVFSearchParams

params = IVFSearchParams()
