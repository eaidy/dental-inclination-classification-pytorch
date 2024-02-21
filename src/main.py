import cnn_model as model

ic_model = model.ICModel()
ic_model.train(dataset_dir='../dataset_2d', epochs=8, batch_size=1)

test_correct, test_losses = ic_model.test(dataset_dir='../dataset_2d_test', batch_size=1)
print(test_correct, test_losses)