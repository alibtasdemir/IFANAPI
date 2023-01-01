import collections

class baseModel():
    def __init__(self, config):
        self.config = config
        self.is_train = config.is_train

        self.network = None
        self.results = collections.OrderedDict()

        self.schedulers = []
        self.optimizers = []

    def get_network(self):
        return self.network

    def eval(self):
        self.network.eval()
