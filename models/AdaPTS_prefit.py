from torch import nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.encoder = nn.Linear(configs.input_channel, configs.output_channel)
        self.decoder = nn.Linear(configs.output_channel, configs.input_channel)
        self.revin = configs.test_with_revin

    def forward(self, batch_x):
        return self.decoder(self.encoder(batch_x))
