import pandas as pd

# Načtení datových sad
df_iscx = pd.read_csv("outs/ex4-iscx.csv", sep=";")
df_mobile = pd.read_csv("outs/ex4-mobile.csv", sep=";")

# Datové sady k zpracování
datasets = {
    "ISCX": df_iscx,
    "Mobile": df_mobile,
}

# Generování LaTeX tabulek
for nazev, df in datasets.items():
    # Přejmenování metod
    df["Metoda"] = df["is_comb"].map({True: "JA4+JA4S+SNI", False: "JA4"})

    # Vytvoření pivot tabulky: sliding_window_size jako řádky, metody jako sloupce
    pivot = df.pivot_table(
        index="sliding_window_size",
        columns="Metoda",
        values="accuracy_overall",
        aggfunc="first",  # nebo např. 'mean' pokud by bylo více hodnot na kombinaci
    ).reset_index()

    # Zaokrouhlení hodnot pro čitelnější výstup
    pivot = pivot.round(4)

    # Export do LaTeX
    latex_code = pivot.to_latex(
        index=False,
        column_format="|c|c|c|",
        header=True,
        escape=False,
        caption=f"Přesnost podle velikosti sliding okna ({nazev})",
        label=f"tab:{nazev.lower()}_acc_vs_window",
    )

    print(latex_code)
