import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from lib.tfutil.ann import build_ann
from lib.dataset.tf_dataset import load_mnist
from lib.dataset.utils import onehot_encoding, get_classes

def test_load_mnist():
    x_train, y_train, x_test, y_test = load_mnist()
    assert x_train.shape == (60000, 28, 28)
    assert y_train.shape == (60000, )
    assert x_test.shape == (10000, 28, 28)
    assert y_test.shape == (10000, )
    assert get_classes(y_test, False) == 10

def test_ann():
    x_train, y_train, x_test, y_test = load_mnist()
    model = build_ann(x_train.shape[1:], get_classes(y_test, False))
    assert model(x_train[0:1]).shape == (1, 10)
