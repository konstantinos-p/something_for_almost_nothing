from utils.train import train_network
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def get_conf_and_train(cfg):
    res = train_network(cfg)
    return res


if __name__ == '__main__':
    get_conf_and_train()
