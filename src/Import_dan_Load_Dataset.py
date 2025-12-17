!pip install ucimlrepo

from ucimlrepo import fetch_ucirepo

acute_inflammations = fetch_ucirepo(id=184)

features = acute_inflammations.data.features
targets = acute_inflammations.data.targets

print(acute_inflammations.metadata)
print(acute_inflammations.variables)