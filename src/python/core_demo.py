import numpy as np
import core_init

# --------------------------
# Item features (rows = items, columns = features)
# Features: Sweet, Bitter, Crunchy, Chocolate, Caffeine
# --------------------------
items = np.array([
    [0, 1, 0, 0, 1],  # Tea
    [0, 1, 0, 0, 1],  # Coffee
    [1, 0, 1, 1, 0],  # Jaffa Cake
    [1, 0, 1, 0, 0],  # Biscuit
    [1, 0, 1, 1, 1],  # Chocolate Bar
    [0, 1, 0, 0, 1],  # Espresso
])

# Names of items
item_names = ["Tea", "Coffee", "Jaffa Cake",
              "Biscuit", "Chocolate Bar", "Espresso"]

# --------------------------
# Example user preference vector
# User likes sweet + chocolate
# --------------------------
user = np.array([[0, 0, 0, 1, 0]])

# Compute cosine similarity
res = core_init.cosine_similarity(user, items)
sim = res["similarity_matrix"]

# Rank items by similarity
top_indices = np.argsort(sim[0])[::-1]  # descending

print("Top recommendations for user:")
for i in top_indices:
    print(f"{item_names[i]}: {sim[0][i]:.2f}")
