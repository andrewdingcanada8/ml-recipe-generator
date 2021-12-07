import json
import os
import re
import parse_recipe as RecipeParser

with open("./recipes_raw/Noodles.mmf", encoding="cp437") as file:
    data = file.read().strip().split("-------- Recipe via Meal-Master (tm) v8.02")
    print(len(data))
    x = (("-------- Recipe via Meal-Master (tm) v8.02" + data[4]).strip()).split("\n")
    # print(x)
    recipe = RecipeParser.parse_recipe(x)
    print(recipe.directions)
