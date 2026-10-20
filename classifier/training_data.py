"""Date de antrenare pentru intent classifier: (query, label).
3 intenții: search / extract / summarize.
"""

TRAIN_DATA: list[tuple[str, str]] = [
    # --- search: caută o informație punctuală ---
    ("Cât costă retragerea de la ATM la pachetul PREMIUM?", "search"),
    ("Ce comision lunar are pachetul STUDENT FREE?", "search"),
    ("Care e costul unui transfer în altă bancă?", "search"),
    ("Cât plătesc pentru emiterea cardului la pachetul INFINITE?", "search"),
    ("Ce taxă are retragerea de numerar la ghișeu?", "search"),
    ("Cât costă pachetul GRIJA COMPLETA pe lună?", "search"),
    ("Care este comisionul pentru plăți online?", "search"),
    ("Ce cost are administrarea contului la SMART?", "search"),
    ("Cât e comisionul la depunerea de numerar?", "search"),
    ("Care sunt taxele pentru pachetul PENSIONAR?", "search"),
    ("Cât costă un ordin de plată programat?", "search"),
    ("Ce preț are schimbul valutar la CEC?", "search"),

    # --- extract: extrage/listează date structurate ---
    ("Listează toate pachetele de cont ale CEC.", "extract"),
    ("Extrage comisioanele pentru pachetul PREMIUM.", "extract"),
    ("Dă-mi toate taxele din pachetul STUDENT FREE.", "extract"),
    ("Enumeră serviciile incluse în pachetul INFINITE.", "extract"),
    ("Care sunt toate comisioanele pentru retrageri?", "extract"),
    ("Extrage lista de pachete și prețurile lor lunare.", "extract"),
    ("Arată-mi toate câmpurile din document.", "extract"),
    ("Dă-mi un tabel cu pachetele și comisioanele lor.", "extract"),
    ("Listează comisioanele pentru transferuri.", "extract"),
    ("Extrage toate taxele de administrare.", "extract"),
    ("Enumeră cardurile disponibile pe fiecare pachet.", "extract"),
    ("Dă-mi toate valorile comisioanelor din document.", "extract"),

    # --- summarize: rezumat ---
    ("Rezumă documentul cu comisioane CEC.", "summarize"),
    ("Pe scurt, ce oferă pachetul PREMIUM?", "summarize"),
    ("Fă-mi un rezumat al pachetului STUDENT FREE.", "summarize"),
    ("Explică-mi pe scurt diferențele dintre pachete.", "summarize"),
    ("Dă-mi un sumar al taxelor principale.", "summarize"),
    ("Rezumă ce conține pachetul INFINITE.", "summarize"),
    ("Pe scurt, care sunt avantajele pachetului GRIJA COMPLETA?", "summarize"),
    ("Fă un rezumat general al documentului.", "summarize"),
    ("Explică pe scurt structura comisioanelor.", "summarize"),
    ("Dă-mi pe scurt ideea principală a documentului.", "summarize"),
    ("Rezumă ofertele CEC pentru pensionari.", "summarize"),
    ("Sumarizează pachetele pentru tineri.", "summarize"),
]