import numpy as np

# Pfade zu den physikochemischen Daten
base_path = "~/BA/BA_ZHAW/data/physicoProperties/ssl_paired/allele"
file_paths = [
    os.path.expanduser(f"{base_path}/Epitope_physico.npz"),
    os.path.expanduser(f"{base_path}/TRA_CDR3_physico.npz"),
    os.path.expanduser(f"{base_path}/TRB_CDR3_physico.npz"),
]

for file_path in file_paths:
    # Lade die Datei
    data = np.load(file_path)
    
    # Zeige einige der berechneten Werte an
    print(f"Datei: {file_path}")
    for key in list(data.keys())[:5]:  # Die ersten 5 Einträge zur Ansicht
        print(f"Sequenz: {key}")
        print(f"Physikochemische Merkmale: {data[key]}")
    print("\n" + "-"*50 + "\n")
