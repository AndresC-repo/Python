"""
Loading yaml files of configuration and can also be used to create folders for test ouputs
"""
import os
import argparse
import yaml
# from datetime import datetime

import torch


class manage_config():
    # def __init__(self):

    def parse_arguments(self,):
        parser = argparse.ArgumentParser()
        parser.add_argument('config', type=str, help="dir to config file")
        args = parser.parse_args()
        return args

    # ----------------------------------------- #

    def get_config(self, config_file):
        with open(config_file, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)
        return cfg

    # ----------------------------------------- #


class config_saver():
    def create_directory(self, config):
        path = config["saver"]
        # now = datetime.now()
        # now = now.strftime("%d:%m:%Y-%H:%M")
        # os.makedirs(path, exist_ok=True)
        save = False
        number = 1
        while not save:
            now = "version_" + str(number)
            path_new = os.path.join(path, now)
            if not os.path.exists(path_new):
                os.makedirs(path_new)
                save = True
            else:
                number = number + 1
        return path_new

    # ----------------------------------------- #

    def save_config(self, config, path):
        file = open(path + "/config.yaml", "w")
        yaml.safe_dump(config, file)

    # ----------------------------------------- #

    def save_model(self, model, path):
        name = "prueba.pt"
        file_name = os.path.join(path, name)
        torch.save(model.state_dict(), file_name)
    # ---------------------------------------------------------------------------------------------- #

    def print_cf(self, name, hp, stamp, conf_mat, detect):
        f = open(name, "a")
        f.write(str(hp))
        f.write("\n")
        f.write(str(stamp))
        f.write("\nconfusion matrix for" + detect + "\n")
        f.write(str(conf_mat))
        f.close()
    # ---------------------------------------------------------------------------------------------- #


# ----------------------------------------- #
#       MAIN
# ----------------------------------------- #


def main(config):
    # pdb.set_trace()
    print('---------')
    '''
    path = create_directory(config)
    save_config(config, path)
    save_model(model, path)
    '''


# ----------------------------------------- #
#   for testing
# ----------------------------------------- #

if __name__ == '__main__':

    # path = '../config/training/config.yaml'
    temp = manage_config()
    args = temp.parse_arguments()
    config = temp.get_config(args.config)
    main(config)
