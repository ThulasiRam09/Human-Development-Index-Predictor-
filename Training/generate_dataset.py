"""
Generates Dataset/HDI.csv
Country-level values (Life expectancy, Mean years of schooling, GNI per capita,
Internet users %) are approximate figures broadly consistent with recent UNDP
Human Development Report statistics. HDI is then computed with the official
UNDP formula (geometric mean of the health / education / income indices),
which is what makes the dataset internally consistent and good for regression.
"""
import numpy as np
import pandas as pd

# Country: (Life expectancy, Mean years of schooling, GNI per capita PPP$, Internet users %)
countries = {
    "Norway": (83.3, 13.1, 82000, 99),
    "Switzerland": (84.0, 13.9, 78000, 96),
    "Iceland": (82.7, 13.9, 68000, 99),
    "Denmark": (81.9, 13.0, 68000, 98),
    "Sweden": (83.3, 12.7, 60000, 96),
    "Germany": (81.4, 14.3, 61000, 93),
    "Australia": (83.9, 12.9, 55000, 91),
    "Netherlands": (82.5, 12.6, 65000, 97),
    "Ireland": (82.6, 12.7, 88000, 92),
    "Hong Kong": (85.5, 12.2, 62000, 92),
    "Finland": (82.1, 12.9, 51000, 95),
    "Belgium": (82.1, 12.4, 57000, 92),
    "Canada": (82.7, 13.9, 51000, 94),
    "Austria": (81.7, 12.5, 61000, 90),
    "New Zealand": (82.5, 13.0, 43000, 93),
    "Luxembourg": (82.5, 12.9, 90000, 98),
    "United Kingdom": (81.3, 13.4, 52000, 96),
    "United States": (77.4, 13.7, 76000, 92),
    "Singapore": (83.6, 11.9, 98000, 92),
    "Japan": (84.7, 13.4, 45000, 90),
    "Korea (Republic of)": (84.0, 12.5, 47000, 97),
    "France": (82.5, 12.2, 52000, 88),
    "Italy": (83.4, 10.7, 47000, 82),
    "Spain": (83.3, 10.3, 44000, 94),
    "Israel": (82.7, 13.2, 44000, 90),
    "Slovenia": (81.6, 12.7, 42000, 88),
    "Czechia": (79.4, 13.0, 41000, 87),
    "Malta": (83.1, 12.4, 46000, 89),
    "Cyprus": (81.5, 12.5, 43000, 92),
    "Estonia": (78.9, 13.4, 42000, 91),
    "United Arab Emirates": (79.0, 12.1, 68000, 99),
    "Qatar": (80.3, 10.4, 92000, 100),
    "Saudi Arabia": (77.5, 10.6, 47000, 100),
    "Poland": (78.6, 12.9, 38000, 88),
    "Lithuania": (76.9, 13.5, 43000, 86),
    "Croatia": (78.9, 12.2, 34000, 82),
    "Greece": (81.8, 10.5, 30000, 82),
    "Portugal": (81.8, 9.6, 36000, 82),
    "Chile": (79.9, 11.1, 26000, 89),
    "Slovakia": (77.8, 12.7, 33000, 84),
    "Hungary": (76.4, 12.1, 37000, 87),
    "Bahrain": (79.5, 10.4, 43000, 99),
    "Latvia": (75.6, 13.6, 34000, 90),
    "Argentina": (75.4, 11.2, 24000, 87),
    "Kuwait": (79.1, 8.0, 60000, 99),
    "Montenegro": (77.0, 12.0, 23000, 82),
    "Romania": (76.1, 11.6, 33000, 86),
    "Russia": (73.0, 12.8, 32000, 88),
    "Bulgaria": (75.4, 11.4, 26000, 78),
    "Kazakhstan": (74.9, 11.9, 26000, 90),
    "Uruguay": (78.0, 9.1, 25000, 88),
    "Bahamas": (73.9, 12.8, 32000, 84),
    "Turkey": (77.5, 8.3, 34000, 84),
    "Panama": (76.4, 10.3, 30000, 74),
    "Malaysia": (76.2, 10.6, 29000, 97),
    "Costa Rica": (78.0, 8.8, 20000, 90),
    "Mexico": (70.2, 9.2, 20000, 76),
    "Serbia": (74.3, 11.2, 20000, 82),
    "Georgia": (73.6, 13.1, 18000, 79),
    "Thailand": (79.3, 8.7, 18000, 88),
    "Brazil": (73.4, 8.3, 16000, 84),
    "China": (78.6, 8.1, 19000, 76),
    "Peru": (75.8, 9.7, 13000, 72),
    "Colombia": (76.3, 9.0, 15000, 71),
    "Ecuador": (74.2, 9.1, 11000, 76),
    "Armenia": (72.6, 11.5, 15000, 79),
    "Albania": (76.4, 10.6, 16000, 83),
    "North Macedonia": (74.8, 10.0, 17000, 84),
    "Iran": (74.6, 10.5, 15000, 78),
    "Dominican Republic": (73.5, 8.4, 19000, 87),
    "Sri Lanka": (77.0, 10.7, 13000, 57),
    "Jamaica": (71.4, 10.0, 9000, 65),
    "Tunisia": (73.7, 7.4, 11000, 71),
    "South Africa": (62.3, 11.0, 14000, 72),
    "Jordan": (74.5, 10.9, 10000, 90),
    "Indonesia": (68.0, 8.6, 12000, 66),
    "Egypt": (70.9, 7.5, 12000, 72),
    "Philippines": (69.3, 9.4, 9000, 68),
    "Viet Nam": (73.7, 8.5, 8000, 78),
    "Bolivia": (65.5, 9.0, 8000, 55),
    "Mongolia": (69.9, 10.3, 12000, 82),
    "Uzbekistan": (71.6, 11.6, 8000, 76),
    "Guatemala": (68.5, 6.7, 8000, 45),
    "Morocco": (74.6, 5.9, 8000, 84),
    "Paraguay": (68.6, 8.9, 13000, 78),
    "Ghana": (64.6, 7.6, 5900, 68),
    "Kenya": (63.6, 6.8, 4500, 32),
    "Namibia": (61.5, 7.2, 9000, 55),
    "Bangladesh": (72.4, 6.2, 6000, 40),
    "Nepal": (69.4, 5.5, 4000, 34),
    "Cambodia": (69.6, 5.2, 4900, 48),
    "Laos": (65.2, 5.3, 7500, 41),
    "India": (67.7, 6.6, 6900, 46),
    "Zambia": (61.2, 7.5, 3500, 32),
    "Tanzania": (66.4, 6.1, 2700, 25),
    "Rwanda": (66.6, 5.4, 2200, 27),
    "Cameroon": (61.2, 6.4, 3900, 38),
    "Zimbabwe": (60.7, 8.7, 2600, 34),
    "Senegal": (68.0, 3.4, 3600, 46),
    "Uganda": (63.2, 5.7, 2200, 25),
    "Cote d'Ivoire": (59.3, 5.3, 5700, 36),
    "Pakistan": (66.1, 4.5, 5500, 21),
    "Myanmar": (65.4, 5.0, 4400, 44),
    "Papua New Guinea": (65.0, 4.9, 4300, 20),
    "Haiti": (64.4, 5.6, 2900, 40),
    "Nigeria": (55.4, 7.2, 5300, 32),
    "Sudan": (65.4, 3.8, 3800, 30),
    "Ethiopia": (66.6, 3.3, 2500, 17),
    "Malawi": (63.8, 5.1, 1500, 15),
    "Congo": (61.5, 6.6, 3200, 21),
    "Benin": (60.7, 3.3, 3400, 28),
    "Guinea": (58.8, 2.6, 2900, 31),
    "Angola": (62.3, 5.4, 6300, 32),
    "Togo": (61.6, 5.0, 2000, 30),
    "Djibouti": (63.7, 4.4, 5900, 33),
    "Mozambique": (59.3, 3.5, 1400, 20),
    "Mali": (59.3, 2.3, 2200, 34),
    "Burkina Faso": (61.6, 1.6, 2200, 22),
    "Chad": (53.3, 2.6, 1700, 15),
    "Niger": (61.6, 2.1, 1300, 10),
    "Yemen": (63.8, 3.2, 1500, 27),
    "Sierra Leone": (61.1, 4.0, 1900, 22),
    "Burundi": (63.7, 3.4, 900, 12),
    "South Sudan": (55.7, 5.5, 900, 8),
    "Central African Republic": (54.5, 4.5, 900, 10),
    "Afghanistan": (62.9, 3.9, 1500, 18),
    "Somalia": (57.5, 1.6, 1300, 8),
    "Liberia": (61.2, 4.7, 1500, 13),
    "Gambia": (63.2, 4.5, 2100, 20),
    "Lesotho": (54.2, 6.9, 3200, 46),
    "Comoros": (64.4, 4.3, 3100, 12),
    "Solomon Islands": (69.9, 5.6, 2400, 15),
    "Vanuatu": (70.6, 7.1, 3000, 27),
    "Fiji": (67.7, 11.1, 12000, 62),
    "Botswana": (61.2, 9.6, 17000, 68),
    "Gabon": (66.4, 9.2, 15000, 62),
    "Eswatini": (57.7, 7.3, 8600, 55),
    "Algeria": (77.1, 8.3, 12000, 72),
    "Libya": (72.1, 7.6, 16000, 45),
    "Suriname": (71.7, 10.4, 13000, 63),
    "Belize": (72.3, 9.9, 9000, 46),
    "El Salvador": (71.1, 7.2, 9200, 61),
    "Honduras": (70.5, 7.0, 6000, 40),
    "Nicaragua": (74.3, 7.1, 6000, 41),
}

rng = np.random.default_rng(42)


def hdi_from_components(life_exp, mys, gni):
    le_index = (life_exp - 20) / (85 - 20)
    edu_index = min(mys / 15, 1.0)
    income_index = (np.log(gni) - np.log(100)) / (np.log(75000) - np.log(100))
    income_index = min(max(income_index, 0), 1)
    hdi = (le_index * edu_index * income_index) ** (1 / 3)
    return round(hdi, 3)


rows = []
for idx, (country, (le, mys, gni, net)) in enumerate(sorted(countries.items()), start=1):
    hdi = hdi_from_components(le, mys, gni)
    # small realistic noise so the model has something to learn, not a perfect formula fit
    hdi = round(min(max(hdi + rng.normal(0, 0.006), 0.25), 0.99), 3)
    rows.append({
        "Id": idx,
        "Country": country,
        "Life expectancy": le,
        "Mean years of schooling": mys,
        "Gross national income (GNI) per capita": gni,
        "Internet users": net,
        "HDI": hdi,
    })

df = pd.DataFrame(rows)
df = df.sort_values("HDI", ascending=False).reset_index(drop=True)
df.insert(0, "HDI Rank", range(1, len(df) + 1))
df.insert(0, "Unnamed: 0", range(0, len(df)))

# introduce a handful of missing values, matching the PDF's description of null handling
for col in ["Life expectancy", "Mean years of schooling", "Gross national income (GNI) per capita", "Internet users"]:
    na_idx = rng.choice(df.index, size=3, replace=False)
    df.loc[na_idx, col] = np.nan

df.to_csv("../Dataset/HDI.csv", index=False)
print(f"Wrote {len(df)} rows to Dataset/HDI.csv")
print(df.head())
