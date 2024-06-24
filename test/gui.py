import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


from lib.dataset.tf_dataset import load_mnist
from lib.dataset.show import show

def test_show():
    x_train, y_train, x_test, y_test = load_mnist()
    show(x_train[0], True)
