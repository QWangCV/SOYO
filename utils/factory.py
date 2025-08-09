from methods.soyo import SOYO

def get_model(model_name, args):
    name = model_name.lower()
    options = {
        'soyo': SOYO,
        }
    return options[name](args)

