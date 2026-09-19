import pandas as pd
import numpy as np

np.random.seed(42)

number_of_records = 500

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

meals = [
    "Breakfast",
    "Lunch",
    "Dinner"
]

menus = [
    "Rice_Dal",
    "Roti_Veg",
    "Biryani",
    "Paneer_Rice",
    "Poha",
    "Chole_Rice"
]

data = []

for i in range(number_of_records):

    day = np.random.choice(days)
    meal = np.random.choice(meals)
    menu = np.random.choice(menus)

    students_present = np.random.randint(80, 501)

    # Different menus have different average waste levels
    menu_waste_factor = {
        "Rice_Dal": 0.06,
        "Roti_Veg": 0.05,
        "Biryani": 0.035,
        "Paneer_Rice": 0.045,
        "Poha": 0.07,
        "Chole_Rice": 0.055
    }

    # Different meals have different waste patterns
    meal_factor = {
        "Breakfast": 1.0,
        "Lunch": 1.2,
        "Dinner": 1.1
    }

    base_waste = (
        students_present
        * menu_waste_factor[menu]
        * meal_factor[meal]
    )

    random_variation = np.random.normal(0, 2.5)

    leftover_kg = base_waste + random_variation + 2

    leftover_kg = max(leftover_kg, 0.5)

    data.append([
        day,
        meal,
        menu,
        students_present,
        round(leftover_kg, 2)
    ])


df = pd.DataFrame(
    data,
    columns=[
        "day",
        "meal",
        "menu",
        "students_present",
        "leftover_kg"
    ]
)

df.to_csv("food_waste_data.csv", index=False)

print("Dataset created successfully!")
print("Number of records:", len(df))
print(df.head())