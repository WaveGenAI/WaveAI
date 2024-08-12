import yaml


class Config:
    """
    Configuration class for the model
    """

    def __init__(self, settings=None):
        if settings is not None:
            self.__dict__.update(self._convert_dict_to_object(settings))
        else:
            self.setup()

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)

    def setup(self, config_path: str = "config.yaml"):
        """Setup the configuration by reading yaml file"""

        with open(config_path, "r", encoding="utf-8") as f:
            settings = yaml.safe_load(f)

        # Recursively update the settings
        self.__dict__.update(self._convert_dict_to_object(settings))

    def _convert_dict_to_object(self, settings):
        """
        Recursively converts a dictionary into an object.
        """
        for key, value in settings.items():
            if isinstance(value, dict):
                settings[key] = Config(value)
        return settings
