import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error

test_df = pd.read_csv("test_output/for_model_test.csv", sep='\t')


test_X = test_df.drop(columns=["Age"])
test_y = test_df["Age"]

pickle_name = "models/histgradient_model.sav"

model = pickle.load(open(pickle_name, 'rb'))

result = model.score(test_X, test_y)

pred_y = model.predict(test_X)
df = pd.DataFrame({    "y_true": test_y,    "y_pred": pred_y,})
df["Sample"] = test_X["Sample"]
df["diff"] = df["y_pred"] - df["y_true"]
print("MAE", mean_absolute_error(test_y, pred_y))

print(result)
breakpoint()
