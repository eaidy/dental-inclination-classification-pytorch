from constants import MODELS

from src.models import cnn_model as model

used_model = MODELS.SELF

train_dataset_absolute_path = '../dataset_2d'
test_dataset_absolute_path = '../dataset_2d_test'

if used_model == MODELS.SELF:
    ic_model = model.ICModel(out_features=3)
    ic_model.train(dataset_dir=train_dataset_absolute_path, epochs=5, batch_size=8, learning_rate=1e-5)
    test_correct, test_losses = ic_model.test(dataset_dir=test_dataset_absolute_path, batch_size=1)
    print(f'Test Correct : {test_correct}\nTest Losses : {test_losses}')

elif used_model == MODELS.VGG16:
    print("sadf")
elif used_model == MODELS.VGG19:
    print("sadf")
elif used_model == MODELS.UNET:
    print("sadf")
elif used_model == MODELS.EFF_NET:
    print("sadf")
elif used_model == MODELS.RES_NET:
    print("sadf")
