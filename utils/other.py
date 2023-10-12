# import pyyaml module
import yaml
from yaml.loader import SafeLoader
import os
import sys

def find_best_hyperparameters(path):
    """
    Find the best hyperparameters for all setups in a given folder.
    Returns
    -------

    """
    os.chdir(path)
    subdirectories = os.listdir()
    unique_directories = []

    try:
        subdirectories.remove('.DS_Store')
    except:
        print('No .DS_STORE file was present')

    for i in range(len(subdirectories)):
        striped_dir_name = ''.join((x for x in subdirectories[i] if not x.isdigit()))
        unique_directories.append(striped_dir_name)
    unique_directories = set(unique_directories)

    dict_results = {}
    for unique_directory in unique_directories:
        dict_results[unique_directory] = [float('inf'),'']

    for dirs in subdirectories:
        try:
            with open(dirs+'/optimization_results.yaml') as f:
                data = yaml.load(f, Loader=SafeLoader)
            striped_dir_name = ''.join((x for x in dirs if not x.isdigit()))
            if dict_results[striped_dir_name][0] > data['best_value']:
                dict_results[striped_dir_name][0] = data['best_value']
                dict_results[striped_dir_name][1] = dirs
        except:
            print('Could not process item: '+dirs)

    for entry in dict_results:
        os.rename(dict_results[entry][1], dict_results[entry][1]+'_best')

    print(dict_results)


if __name__ == '__main__':
    path = sys.argv[1]
    find_best_hyperparameters(path)
