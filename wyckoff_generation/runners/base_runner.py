from wyckoff_generation.common.registry import registry


class BaseRunner:
    def __init__(self, config):
        self.device = config["device"]
        self.config = config
        self.num_epochs = config["epochs"]
        self.val_interval = config["val_interval"]

        self.init_model(config)
        self.init_dataloaders(config)
        self.init_optimizer(config)
        self.init_logger(config)

    def init_model(self, config):
        raise NotImplementedError

    def init_dataloaders(self, config):
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ) = registry.get_dataset_class(config["dataset"]).get_dataloaders(config)

    def init_optimizer(self, config):
        raise NotImplementedError

    def init_logger(self, config):
        self.logger = registry.get_logger_class(config["logger"])(config)

    def load_checkpoint(self, config):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
