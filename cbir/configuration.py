import cbir


def configure(config_path, args):
    """
    :param config_path:
    :param args: May contain some parameters for config.
                 Has higher priority than config from `config_path`
    """
    # TODO: Rewrite.
    if 'database' in args:
        cbir.CONFIG['database'] = args.database
