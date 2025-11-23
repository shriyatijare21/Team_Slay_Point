import random
import pandas as pd

# ----------------------------------------------------------
# 1. EXPANDED CATEGORY → MERCHANTS + KEYWORDS
# ----------------------------------------------------------

CATEGORY_DATA = {
    "Food & Dining": {
        "merchants": [
            "Starbucks", "Dominos", "KFC", "McDonalds", "Cafe Coffee Day",
            "Burger King", "Pizza Hut", "Subway", "Barbeque Nation",
            "Faasos", "Behrouz Biryani", "Bikanervala", "Haldirams",
            "Chayos", "Social", "Mainland China"
        ],
        "keywords": [
            "food", "restaurant", "dining", "cafe", "eatery", "meal",
            "breakfast", "lunch", "dinner", "snacks", "delivery"
        ]
    },
    "Shopping": {
        "merchants": [
            "Amazon", "Flipkart", "Myntra", "Ajio", "H&M", "Zara",
            "Lifestyle", "Max Fashion", "Reliance Trends", "Pantaloons",
            "DMart Fashion", "FirstCry", "Nykaa", "BigBazaar Online",
            "Decathlon"
        ],
        "keywords": [
            "shopping", "store", "mall", "retail", "fashion", "clothes",
            "apparel", "footwear", "beauty", "sale"
        ]
    },
    "Fuel": {
        "merchants": [
            "Indian Oil", "HP Petrol Pump", "BPCL Station", "Shell Fuel",
            "Nayara Energy", "Essar Fuel", "JioBP"
        ],
        "keywords": [
            "fuel", "petrol", "diesel", "gas", "gasoline", "refill",
            "pump", "fuelstation"
        ]
    },
    "Groceries": {
        "merchants": [
            "D-Mart", "Big Bazaar", "Reliance Fresh", "More Retail",
            "Spencers", "Walmart", "BigBasket", "Nature's Basket",
            "FreshMart", "Metro Wholesale"
        ],
        "keywords": [
            "groceries", "grocery", "supermarket", "mart", "foods",
            "household", "vegetables", "fruits", "daily needs"
        ]
    },
    "Bills & Utilities": {
        "merchants": [
            "Airtel", "Jio", "Vodafone", "BSNL", "ACT Fiber",
            "Tata Power", "MSEB Electricity", "Reliance Energy"
        ],
        "keywords": [
            "bill", "electricity", "recharge", "utility", "mobile",
            "postpaid", "prepaid", "broadband"
        ]
    },
    "Travel": {
        "merchants": [
            "Uber", "Ola", "Indigo Airlines", "Air India", "Vistara",
            "IRCTC", "RedBus", "Rapido", "GoAir", "MakeMyTrip"
        ],
        "keywords": [
            "travel", "ride", "cab", "taxi", "flight", "airlines",
            "bus", "train", "booking", "journey"
        ]
    },
    "Entertainment": {
        "merchants": [
            "Netflix", "Spotify", "BookMyShow", "PVR Cinemas",
            "INOX", "Hotstar", "Prime Video", "Gaana", "SonyLiv"
        ],
        "keywords": [
            "entertainment", "movie", "cinema", "subscription",
            "music", "streaming", "series", "tickets"
        ]
    },
    "Rent": {
        "merchants": [
            "Rent Transfer", "Landlord", "House Owner", "PG Rent",
            "Zolo", "Nestaway", "FlatMate Rent"
        ],
        "keywords": [
            "rent", "lease", "house rent", "flat rent", "monthly rent"
        ]
    }
}

# ----------------------------------------------------------
# 2. NOISE / MISSPELLINGS / RANDOM MODIFIERS
# ----------------------------------------------------------

def add_noise(text):
    noise_patterns = [
        lambda x: x + " Pvt Ltd",
        lambda x: x.replace(" ", ""),
        lambda x: x.upper(),
        lambda x: x.lower(),
        lambda x: x + " #" + str(random.randint(10, 999)),
        lambda x: x.replace("a", "@").replace("o", "0"),
        lambda x: x + f" {random.randint(1, 50)}",
        lambda x: x + " corp",
        lambda x: "the " + x,
        lambda x: x + " india",
    ]
    return random.choice(noise_patterns)(text)


# ----------------------------------------------------------
# 3. GENERATE RAW STRING COMBINATIONS
# ----------------------------------------------------------

def generate_variation(merchant, keyword):
    formats = [

        # pure merchant
        merchant,

        # merchant + noise
        add_noise(merchant),

        # keyword only
        keyword,
        keyword + " payment",
        keyword + " service",

        # keyword + noise
        add_noise(keyword),

        # merchant + keyword
        f"{merchant} {keyword}",
        f"{keyword} {merchant}",
        f"{merchant} - {keyword}",
        f"{keyword} at {merchant}",
        f"{merchant} {keyword} txn",

        # merchant + noise + keyword
        f"{add_noise(merchant)} {keyword}",
        f"{keyword} {add_noise(merchant)}",

        # corrupted noisy patterns
        merchant[:3].lower() + keyword[:3].lower(),
        keyword[:4] + merchant[:2],
        merchant.lower().replace(" ", "") + keyword[0:2],
    ]

    return random.choice(formats)


# ----------------------------------------------------------
# 4. GENERATE DATASET
# ----------------------------------------------------------

def generate_dataset(n=50000, output="transactions.csv"):
    data = []

    for _ in range(n):
        category = random.choice(list(CATEGORY_DATA.keys()))
        cat_info = CATEGORY_DATA[category]

        merchant = random.choice(cat_info["merchants"])
        keyword = random.choice(cat_info["keywords"])

        transaction_string = generate_variation(merchant, keyword)

        data.append({
            "transaction": transaction_string,
            "category": category
        })

    df = pd.DataFrame(data)
    df.to_csv(output, index=False)
    print(f"Generated {len(df)} rows → {output}")
    return df


# ----------------------------------------------------------
# RUN
# ----------------------------------------------------------

if __name__ == "__main__":
    generate_dataset(200000)
