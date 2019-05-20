from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import utils
import numpy
from PIL import Image

object_to_predict = 3


def main():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    y_train = utils.to_categorical(y_train, 10)
    x_train /= 255

    classes = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

    model = load_model("fashion_model.h5")

    pixels = x_train[object_to_predict]
    image = Image.fromarray(pixels, "L")

    prediction = model.predict(numpy.reshape(x_train[object_to_predict], (1, 784)))

    print("Object",
          classes[numpy.argmax(y_train[object_to_predict])],
          "recognised as",
          classes[numpy.argmax(prediction)],
          "with accuracy",
          numpy.max(prediction*100),
          "percents")

    image.show()


if __name__ == "__main__":
    main()
