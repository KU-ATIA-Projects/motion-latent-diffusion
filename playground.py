import json
import os
import time
import logging
from pathlib import Path
from multiprocessing.sharedctypes import Value
from builtins import ValueError

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from torch.utils.data import DataLoader, ConcatDataset
from omegaconf import OmegaConf
from rich.progress import track

from mld.callback import ProgressLogger
from mld.config import parse_args
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.data.utils import a2m_collate
from mld.models.get_model import get_model
from mld.utils.logger import create_logger



def main():
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    logger = create_logger(cfg, phase="demo")

    if cfg.DEMO.EXAMPLE:
        # Check txt file input
        # load txt
        from mld.utils.demo_utils import load_example_input

        text, length = load_example_input(cfg.DEMO.EXAMPLE)
        task = "Example"
    elif cfg.DEMO.TASK:
        task = cfg.DEMO.TASK
        text = None

        # default lengths
        length = 200 if not length else length
        length = [int(length)]
        text = [text]

    output_dir = Path(
        os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
                     "latentspace_" + cfg.TIME))
    output_dir.mkdir(parents=True, exist_ok=True)

    # cuda options
    if cfg.ACCELERATOR == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(x) for x in cfg.DEVICE)
        device = torch.device("cuda")

    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    total_time = time.time()
    model = get_model(cfg, dataset)


    # loading checkpoints
    logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    mld_time = time.time()


    # sample
    with torch.no_grad():
        rep_lst = []    
        rep_ref_lst = []
        texts_lst = []
        # task: input or Example
            # prepare batch data
        batch = {"length": length, "text": text}
        
        for rep in range(cfg.DEMO.REPLICATION):

            joints = model(batch)
            # cal inference time
            infer_time = time.time() - mld_time
            num_batch = 1
            num_all_frame = sum(batch["length"])
            num_ave_frame = sum(batch["length"]) / len(batch["length"])

            # upscaling to compare with other methods
            # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
            nsample = len(joints)
            id = 0
            for i in range(nsample):
                npypath = str(output_dir /
                            f"{task}_{length[i]}_batch{id}_{i}_{rep}.npy")
                with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                    text_file.write(batch["text"][i])
                np.save(npypath, joints[i].detach().cpu().numpy())
                logger.info(f"Motions are generated here:\n{npypath}")
            
            if cfg.DEMO.OUTALL:
                rep_lst.append(joints)
                texts_lst.append(batch["text"])
                
                
        if cfg.DEMO.OUTALL:
            grouped_lst = []
            for n in range(nsample):
                grouped_lst.append(torch.cat([r[n][None] for r in rep_lst], dim=0)[None]) 
            combinedOut = torch.cat(grouped_lst, dim=0)
            try:
                # save all motions
                npypath = str(output_dir / f"{task}_{length[i]}_all.npy")
                
                np.save(npypath,combinedOut.detach().cpu().numpy())
                with open(npypath.replace('npy','txt'),"w") as text_file: 
                    for texts in texts_lst:
                        for text in texts:
                            text_file.write(text)
                            text_file.write('\n')
                logger.info(f"All reconstructed motions are generated here:\n{npypath}")
            except:
                raise ValueError("Lengths of motions are different, so we cannot save all motions in one file.")
                    

    # # random samlping
    # if not text:
    #     if task == "random_sampling":
    #         # default text
    #         text = "random sampling"
    #         length = 196
    #         nsample, latent_dim = 500, 256
    #         batch = {
    #             "latent":
    #             torch.randn(1, nsample, latent_dim, device=model.device),
    #             "length": [int(length)] * nsample,
    #         }
    #         # # n 
    #         # batch_save = batch.copy()
    #         # batch_save["latent"] = batch_save["latent"].cpu().numpy().tolist()
    #         # with open("/home/pjr726/motion-latent-diffusion/results/latent_vector.json", "w") as f:
    #         #     json.dump(batch_save, f)

    #         # vae random sampling
    #         joints = model.gen_from_latent(batch)

    #         # latent diffusion random sampling
    #         # for i in range(100):
    #         #     model.condition = 'text_uncond'
    #         #     joints = model(batch)

    #         num_batch, num_all_frame, num_ave_frame = 100, 100 * 196, 196
    #         infer_time = time.time() - mld_time

    #         # joints = joints.cpu().numpy()

    #         # upscaling to compare with other methods
    #         # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
    #         for i in range(nsample):
    #             npypath = output_dir / \
    #                 f"{text.split(' ')[0]}_{length}_{i}.npy"
    #             np.save(npypath, joints[i].detach().cpu().numpy())
    #             logger.info(f"Motions are generated here:\n{npypath}")



        # ToDo fix time counting
        total_time = time.time() - total_time
        print(f'MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(f'MLD Infer time - This/Ave batch: {infer_time/num_batch:.2f}')
        print(f'MLD Infer FPS - Total batch: {num_all_frame/infer_time:.2f}')
        print(
            f'MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')
        print(
            f'MLD Infer FPS - {num_all_frame/infer_time:.2f}s')
        print(
            f'MLD Infer FPS - Running Poses Per Second: {num_ave_frame*infer_time/num_batch:.2f}')

        # todo no num_batch!!!
        # num_batch=> num_forward
        print(
            f'MLD Infer FPS - time for 100 Poses: {infer_time/(num_batch*num_ave_frame)*100:.2f}'
        )
        print(
            f'Total time spent: {total_time:.2f} seconds (including model loading time and exporting time).'
        )


if __name__ == "__main__":
    main()



















































def data_parse(step: int, latents: np.ndarray, classids: list):
    nsample = 30

    # classids = list(range(0,12))
    nclass = len(classids)
    # (12, 50, 50, 256)
    t_0 = latents[classids,:nsample, step,:]
    t_0 = t_0.reshape(-1, t_0.shape[-1])


    # labels = np.array(list(range(0,nclass)))
    # labels = labels.repeat(nsample)

    # labels = np.array(['sit', 'lift_dumbbell', 'turn_steering'])
    labels = np.array(['throw', 'walk', 'boxing'])
    labels = labels.repeat(nsample)
    # labels = [['sit']* nsample,['lift_dumbbell']* nsample, ['turn steering wheel']* nsample]
    # labels = labels * nsample

    tsne = TSNE(n_components=2, verbose=0, random_state=123)
    z = tsne.fit_transform(t_0) 
    df = pd.DataFrame()

    # normalize
    z = 1.8*(z-np.min(z,axis=0))/(np.max(z,axis=0)-np.min(z,axis=0)) -0.9

    df["y"] = labels
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]
    return df

def drawFig(output_dir: str, latents: np.ndarray, classids: list = [8,6,5], steps: list = [0, 15, 35, 49] ):
    ''' 
    Draw the figure of t-SNE
    Parameters:
        output_dir: output directory
        latents: (12, 50, 50, 256)
        steps: list of diffusion steps to draw
        classids: list of class ids
            # 0: "warm_up",
            # 1: "walk",
            # 2: "run",
            # 3: "jump",
            # 4: "drink",
            # 5: "lift_dumbbell",
            # 6: "sit",
            # 7: "eat",
            # 8: "turn steering wheel",
            # 9: "phone",
            # 10: "boxing",
            # 11: "throw",
    '''

    sns.set()

    fig, axs = plt.subplots(1, 4, figsize=(4*3,2.5))

    nclass = len(classids)
    steps.sort(reverse=True)
    for i, step in enumerate(steps):
        df = data_parse(steps[0]-step,latents,classids)
        sns.scatterplot(ax=axs[i], x="comp-1", y="comp-2", hue='y',
                        legend = False if i != len(steps) -1  else True,
                        palette=sns.color_palette("hls", nclass),
                        data=df).set(title=r"t = {}".format(step)) 

        axs[i].set_xlim((-1, 1))
        axs[i].set_ylim((-1, 1))

    plt.legend(loc=[1.1,0.2], title='Action ID')

    plt.tight_layout()
    plt.savefig(pjoin(output_dir, 'TSNE.png'), bbox_inches='tight')
    plt.show()

# def main():
#     # parse options
#     cfg = parse_args(phase="test")  # parse config file
#     cfg.FOLDER = cfg.TEST.FOLDER
#     # create logger
#     logger = create_logger(cfg, phase="test")
#     output_dir = Path(
#         os.path.join(cfg.FOLDER, str(cfg.model.model_type), str(cfg.NAME),
#                      "latentspace_" + cfg.TIME))
#     output_dir.mkdir(parents=True, exist_ok=True)
#     logger.info(OmegaConf.to_yaml(cfg))

#     # set seed
#     pl.seed_everything(cfg.SEED_VALUE)

#     # gpu setting
#     if cfg.ACCELERATOR == "gpu":
#         # os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
#         #     str(x) for x in cfg.DEVICE)
#         os.environ["PYTHONWARNINGS"] = "ignore"
#         os.environ["TOKENIZERS_PARALLELISM"] = "false"

#     # create dataset
#     dataset = get_datasets(cfg, logger=logger, phase="test")[0]
#     logger.info("datasets module {} initialized".format("".join(
#         cfg.TRAIN.DATASETS)))
#     subset = 'train'.upper() 
#     split = eval(f"cfg.{subset}.SPLIT")
#     split_file = pjoin(
#                     eval(f"cfg.DATASET.{dataset.name.upper()}.SPLIT_ROOT"),
#                     eval(f"cfg.{subset}.SPLIT") + ".txt",
#                 )
#     dataloader = DataLoader(dataset.Dataset(split_file=split_file,split=split,**dataset.hparams),batch_size=8,collate_fn=a2m_collate)

#     # create model
#     model = get_model(cfg, dataset)
#     logger.info("model {} loaded".format(cfg.model.model_type))

#     # loading state dict
#     logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))

#     state_dict = torch.load(cfg.TEST.CHECKPOINTS,
#                             map_location="cpu")["state_dict"]
#     model.load_state_dict(state_dict)
#     model = model.eval()
    
#     # Device
#     if cfg.ACCELERATOR == "gpu":
#         device = torch.device("cuda")
#         model = model.to(device)
    
#     # Generate latent codes
#     with torch.no_grad():
#         labels = torch.tensor(np.array(list(range(0,dataset.nclasses)))).unsqueeze(1).to(device)
#         lengths = torch.tensor([120]*dataset.nclasses).to(device)
#         z_list = []
#         for i in track(range(50),'Generating latent codes'):
#             cond_emb = torch.cat((torch.zeros_like(labels), labels))
#             # [steps, classes, latent_dim]
#             z = model._diffusion_reverse_tsne(cond_emb, lengths)
#             z_list.append(z)
#         # [samples, steps, classes, latent_dim] -> [classes, samples, steps, latent_dim]
#         latents = torch.stack(z_list, dim=0).permute(2,0,1,3).cpu().numpy()
#         print(latents.shape)
#         selected_latents = latents[:,:30,49,:]
#         selected_latents = torch.from_numpy(selected_latents)
#         print(selected_latents.shape)


#         classids = [11, 1, 10]
#         labels = np.array(['throw', 'walk', 'boxing'])

#         # Define nsample and latent_dim as per your setup
#         nsample = 30

#         for idx, classid in enumerate(classids):
#             # Generate the batch for each class
#             batch = {
#             "latent": selected_latents[classid,:,:].unsqueeze(0).to(device=model.device),
#             "length": [120] * nsample,
#             }

#             # Generate the joints from the latent batch
#             joints = model.gen_from_latent(batch)
#             joints = joints.cpu().numpy()

#             # Save the joints to a numpy file
#             for i in range(nsample):
#                 npypath = output_dir / f"{labels[idx]}_120_{i}.npy"
#                 np.save(npypath, joints[0, i])
#                 logger.info(f"Motions are generated here:\n{npypath}")





#         batch = {
#             "latent": selected_latents.to(device=model.device),  # Moving to the desired device
#             "length": [120] * selected_latents.shape[0]
#         }








#         joints = model.gen_from_latent(batch)
#         joints = joints.cpu().numpy()
#         print(joints.shape)


#         for idx, classid in enumerate(classids):
#             # Extract the latent codes for the specific class
#             class_latents = selected_latents[classid]

#             for i in range(30):
#                 # Save the latent codes to a numpy file
#                 npypath = output_dir / f"{labels[idx]}_{lengths}_{i}.npy"
#                 np.save(npypath, class_latents[i].detach().cpu().numpy())
#                 logger.info(f"Motions are generated here:\n{npypath}")

#     # Draw figure
#     # drawFig(output_dir, latents, classids = [11,1,10], steps = [0, 15, 35, 49])
#     logger.info("TSNE figure and npy files saved to {}".format(output_dir))

if __name__ == "__main__":
    main()
