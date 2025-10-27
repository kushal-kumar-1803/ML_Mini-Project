from preprocess import preprocess_data
from models.predictor import build_predictor
import os

os.makedirs('models', exist_ok=True)

X_train, X_test, y_train, y_test = preprocess_data()
model = build_predictor(input_dim=X_train.shape[1])

model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.2)
model.save('models/predictor_mlp.h5')

print("✅ Predictor model saved as models/predictor_mlp.h5")
