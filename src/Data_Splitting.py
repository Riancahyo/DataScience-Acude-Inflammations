#@title Train Test Split
from sklearn.model_selection import train_test_split

X = features_cleaned.copy()
y = target_data['bladder-inflammation'].copy()

# Bagi data menjadi Training, Validation dan Test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Bagi Training + Validation menjadi Training dan Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.25,
    random_state=42,
    stratify=y_train
)

print("Ukuran data Training:", X_train.shape)
print("Ukuran data Validation:", X_val.shape)
print("Ukuran data Testing:", X_test.shape)


