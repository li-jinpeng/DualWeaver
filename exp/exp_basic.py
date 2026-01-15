from models import (
    SFT,
    AdaPTS,
    DualWeaver,
    ZeroShot,
)


class Exp_Basic(object):
    def __init__(self, args):
        print(args)
        self.args = args
        self.adapter_dict = {
            "AdaPTS": AdaPTS,
            "ZeroShot": ZeroShot,
            "WeaverMLP": DualWeaver,
            "WeaverCNN": DualWeaver,
            "SFT": SFT,
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
