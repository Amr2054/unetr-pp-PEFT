import yaml

class Config:
    def __init__(self, config_dict):
        self.paths = config_dict.get("paths", {})
        self.training = config_dict.get("training", {})


    def __repr__(self):
        return f"Config(paths={self.paths}, training={self.training})"

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    config = Config(config)
    return config