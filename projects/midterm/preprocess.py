import pandas as pd

def read_dataset() -> pd.DataFrame:
    df = pd.read_csv("data/datos_3t25/CSV/EPA_2025T3.tab", sep="\t")
    df.columns = df.columns.str.lower()

    return df

def get_prov_values() -> dict:
    return {
        1: "Araba",
        2: "Albacete",
        3: "Alacant",
        4: "Almería",
        5: "Ávila",
        6: "Badajoz",
        7: "Balears, Illes",
        8: "Barcelona",
        9: "Burgos",
        10: "Cáceres",
        11: "Cádiz",
        12: "Castelló",
        13: "Ciudad Real",
        14: "Córdoba",
        15: "Coruña, A",
        16: "Cuenca",
        17: "Girona",
        18: "Granada",
        19: "Guadalajara",
        20: "Gipuzkoa",
        21: "Huelva",
        22: "Huesca",
        23: "Jaén",
        24: "León",
        25: "Lleida",
        26: "Rioja, La",
        27: "Lugo",
        28: "Madrid",
        29: "Málaga",
        30: "Murcia",
        31: "Navarra",
        32: "Ourense",
        33: "Asturias",
        34: "Palencia",
        35: "Palmas, Las",
        36: "Pontevedra",
        37: "Salamanca",
        38: "Santa Cruz de Tenerife",
        39: "Cantabria",
        40: "Segovia",
        41: "Sevilla",
        42: "Soria",
        43: "Tarragona",
        44: "Teruel",
        45: "Toledo",
        46: "València",
        47: "Valladolid",
        48: "Bizkaia",
        49: "Zamora",
        50: "Zaragoza",
        51: "Ceuta",
        52: "Melilla",
    }

def map_prov(df: pd.DataFrame) -> pd.DataFrame:
    return df.prov.map(get_prov_values())

def get_edad1_values() -> dict:
    return {
        0: "0 to 4 years",
        5: "5 to 9 years",
        10: "10 to 15 years",
        16: "16 to 19 years",
        20: "20 to 24 years",
        25: "25 to 29 years",
        30: "30 to 34 years",
        35: "35 to 39 years",
        40: "40 to 44 years",
        45: "45 to 49 years",
        50: "50 to 54 years",
        55: "55 to 59 years",
        60: "60 to 64 years",
        65: "65 or more years",
    }

def map_edad1(df: pd.DataFrame) -> pd.DataFrame:
    return df.edad1.map(get_edad1_values())

def get_sexo1_values() -> dict:
    return {
        1: "Man",
        6: "Woman",
    }

def map_sexo1(df: pd.DataFrame) -> pd.DataFrame:
    return df.sexo1.map(get_sexo1_values())

def get_eciv1_values() -> dict:
    return {
        1: "Single",
        2: "Married",
        3: "Widowed",
        4: "Separated or divorced",
    }

def map_eciv1(df: pd.DataFrame) -> pd.DataFrame:
    return df.eciv1.map(get_eciv1_values())

def get_nforma_values() -> dict:
    return {
        "AN": "Illiterate",
        "P1": "Incomplete primary education",
        "P2": "Primary education",
        "S1": "Lower secondary education",
        "SG": "Upper secondary education — general track",
        "SP": "Upper secondary education — vocational track",
        "SU": "Higher education",
    }

def map_nforma(df: pd.DataFrame) -> pd.DataFrame:
    return df.nforma.map(get_nforma_values())

def get_trarem_values() -> dict:
    return {
        1: "Yes",
        6: "No",
    }

def map_trarem(df: pd.DataFrame) -> pd.DataFrame:
    return df.trarem.map(get_trarem_values())

def reduced_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_reduced = pd.DataFrame()

    df_reduced["prov"] = map_prov(df)
    df_reduced["edad1"] = map_edad1(df)
    df_reduced["sexo1"] = map_sexo1(df)
    df_reduced["eciv1"] = map_eciv1(df)
    df_reduced["nforma"] = map_nforma(df)
    df_reduced["trarem"] = map_trarem(df)

    return df_reduced

def filtered_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_reduced = reduced_dataset(df)

    return df_reduced[df_reduced.trarem.isnull() == False]
