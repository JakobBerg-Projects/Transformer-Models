def get_num_params(model):
    """
    Get the number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_config(config):
    """
    Print the configuration to the console.
    """
    print("Using configuration:")
    for key, value in vars(config).items():
        print(f"\t{key}: {value}")
    print("*" * 100)

