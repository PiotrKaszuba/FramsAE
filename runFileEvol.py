import sys
from Code.Evolution.frams_evolution_latent_slurm import runEvolLatent

if __name__ == "__main__":
    evol_name = sys.argv[1] if sys.argv[1] != 'None' else None
    representation = sys.argv[2] if sys.argv[2] != 'None' else None
    latent = sys.argv[3] if sys.argv[3] != 'None' else None

    data_path = sys.argv[4] if sys.argv[4] != 'None' else None
    load_dir = sys.argv[5] if sys.argv[5] != 'None' else None
    frams_path = sys.argv[6] if sys.argv[6] != 'None' else None
    rand = sys.argv[7] if sys.argv[7] != 'None' else None
    if rand is not None:
        rand = True if rand == 'True' else False
    gene = int(sys.argv[8]) if sys.argv[8] != 'None' else None
    pop = int(sys.argv[9]) if sys.argv[9] != 'None' else None



    runEvolLatent(evol_name=evol_name, representation=representation, data_path=data_path, load_dir=load_dir,
                  frams_path=frams_path, rand=rand, gene=gene, pop_s= pop, latent = latent)
    # runEvol(evol_name='test', representation='f1', data_path='mods', load_dir='',
    #         frams_path='C:/Users/Piotr/Desktop/Framsticks50rc14')