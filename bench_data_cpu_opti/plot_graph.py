import matplotlib.pyplot as plt
import pandas as pd
import os

def get_data():
    for file in os.listdir("."):
        if file.endswith(".csv"):
            df = pd.read_csv(file)
            df['version'] = file.split(".")[0]  # On ajoute une colonne pour identifier l'algorithme
            yield df

all_data = list(get_data())

plt.figure(figsize=(10, 6))
for df in all_data:
    plt.plot(df['N'], df['Time_ms'], marker='o', label=df['version'][0])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nombre de points')
plt.ylabel('Temps (ms)')
plt.title("Comparaison des temps d'exécution (Log-Log)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("time_comparison_plot.png")
plt.show()

plt.figure(figsize=(10, 6))
for df in all_data:
    plt.plot(df['N'], df['Memory_MiB'], marker='o', label=df['version'][0])

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Nombre de points')
plt.ylabel('Mémoire (MiB)')
plt.title('Comparaison de la mémoire utilisée (Log-Log)')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("memory_comparison_plot.png")
plt.show()
