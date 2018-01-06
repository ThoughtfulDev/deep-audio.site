from keras.utils import plot_model
from keras.models import load_model

model = load_model('model.h5')
plot_model(model, to_file='model.png', show_shapes=True)