from models import (
    Finetune_full,
    Finetune_linear_probing,
    WeaverCNN,
    AdaPTS,
    WeaverMLP,
    ZeroShot,
)


class Exp_Basic(object):
    def __init__(self, args):
        print(args)
        self.args = args
        self.adapter_dict = {
            "AdaPTS": AdaPTS,
            "ZeroShot": ZeroShot,
            "WeaverMLP": WeaverMLP,
            "WeaverCNN": WeaverCNN,
            "Finetune_full": Finetune_full,
            "Finetune_linear_probing": Finetune_linear_probing,
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
