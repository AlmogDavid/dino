try:
    from .core import *
    from .modules import *
except (ModuleNotFoundError, ImportError):
    from core import *
    from modules import *
import hydra
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
import seaborn as sns
from collections import defaultdict

torch.multiprocessing.set_sharing_strategy('file_system')


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class CRFComputer(Dataset):

    def __init__(self, dataset, outputs, run_crf):
        self.dataset = dataset
        self.outputs = outputs
        self.run_crf = run_crf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        if "linear_probs" in self.outputs:
            if self.run_crf:
                batch["linear_probs"] = dense_crf(batch["img"].cpu().detach(),
                                                  self.outputs["linear_probs"][item].cpu().detach())
                batch["cluster_probs"] = dense_crf(batch["img"].cpu().detach(),
                                                   self.outputs["cluster_probs"][item].cpu().detach())
            else:
                batch["linear_probs"] = self.outputs["linear_probs"][item]
                batch["cluster_probs"] = self.outputs["cluster_probs"][item]
        if "picie_preds" in self.outputs:
            batch["picie_preds"] = self.outputs["picie_preds"][item]
        return batch


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    #print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir


    # Best
    model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/cocostuff27_vit_base_5.ckpt")
    #model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/cityscapes_vit_base_1.ckpt")

    print(OmegaConf.to_yaml(model.cfg))

    compare_to_picie = False
    if compare_to_picie:
        picie_state = torch.load("../saved_models/picie_and_probes.pth")
        picie = picie_state["model"]
        picie_cluster_probe = picie_state["cluster_probe"]
        picie_cluster_metrics = picie_state["cluster_metrics"]

    run_crf = False
    run_prediction = False


    if cfg.dataset_name == "voc":
        loader_crop = None
    else:
        loader_crop = "center"

    test_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=model.cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, loader_crop),
        target_transform=get_transform(320, True, loader_crop),
        cfg=model.cfg,
    )

    test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True, collate_fn=flexible_collate)

    outputs = defaultdict(list)
    model.eval().cuda()
    par_model = torch.nn.DataParallel(model.net)
    if compare_to_picie:
        par_picie = torch.nn.DataParallel(picie)

    if run_prediction:
        for i, batch in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                img = batch["img"].cuda()
                label = batch["label"].cuda()\

                feats, code1 = par_model(img)
                feats, code2 = par_model(img.flip(dims=[3]))
                code = (code1 + code2.flip(dims=[3])) / 2

                code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                outputs["linear_probs"].append(torch.log_softmax(model.linear_probe(code), dim=1).cpu())
                outputs["cluster_probs"].append(model.cluster_probe(code, 2, log_probs=True).cpu())

                if compare_to_picie:
                    outputs["picie_preds"].append(picie_cluster_metrics.map_clusters(
                        picie_cluster_probe(par_picie(img), None)[1].argmax(1).cpu()))
    outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

    if run_crf:
        crf_batch_size = 5
    else:
        crf_batch_size = cfg.batch_size * 2
    crf_dataset = CRFComputer(test_dataset, outputs, run_crf)
    crf_loader = DataLoader(crf_dataset, crf_batch_size,
                            shuffle=False, num_workers=cfg.num_workers + 5,
                            pin_memory=True, collate_fn=flexible_collate)

    crf_outputs = defaultdict(list)
    for i, batch in enumerate(tqdm(crf_loader)):
        with torch.no_grad():
            label = batch["label"].cuda(non_blocking=True)
            img = batch["img"]
            if run_prediction:
                linear_preds = batch["linear_probs"].cuda(non_blocking=True).argmax(1)
                cluster_preds = batch["cluster_probs"].cuda(non_blocking=True).argmax(1)
                model.test_linear_metrics.update(linear_preds, label)
                model.test_cluster_metrics.update(cluster_preds, label)
                crf_outputs['linear_preds'].append(linear_preds[:model.cfg.n_images].detach().cpu())
                crf_outputs["cluster_preds"].append(cluster_preds[:model.cfg.n_images].detach().cpu()),

            if compare_to_picie:
                crf_outputs["picie_preds"].append(batch["picie_preds"])

            crf_outputs["img"].append(img[:model.cfg.n_images].detach().cpu())
            crf_outputs["label"].append(label[:model.cfg.n_images].detach().cpu())
    crf_outputs = {k: torch.cat(v, dim=0) for k, v in crf_outputs.items()}

    tb_metrics = {
        **model.test_linear_metrics.compute(),
        **model.test_cluster_metrics.compute(),
    }

    print("")
    print(tb_metrics)

    # good_images = [19, 50, 54, 67, 66, 65, 75, 77, 76, 124]
    if model.cfg.dataset_name == "cocostuff27":
        all_good_images = range(20)
        #all_good_images = [61, 60, 49, 44, 13, 70]
        #all_good_images = [19, 54, 67, 66, 65, 75, 77, 76, 124]
    elif model.cfg.dataset_name == "cityscapes":
        #all_good_images = range(80)
        #all_good_images = [ 5, 20, 56]
        all_good_images = [11, 32, 43, 52]
    else:
        raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))

    # good_images = range(10)

    # for start in range(0, 200, 5):

    if run_prediction:
        n_rows = 3
    else:
        n_rows = 2

    if compare_to_picie:
        n_rows += 1

    plt.rcParams['figure.facecolor'] = 'black'

    for good_images in batch_list(all_good_images, 3):
        fig, ax = plt.subplots(n_rows, len(good_images), figsize=(len(good_images) * 3, n_rows * 3))
        for i, img_num in enumerate(good_images):
            # img_num = start + i
            ax[0, i].imshow(prep_for_plot(crf_outputs["img"][img_num]))
            ax[1, i].imshow(model.label_cmap[crf_outputs["label"][img_num]])
            # ax[2, i].imshow(model.label_cmap[crf_outputs["linear_preds"][img_num]])
            if run_prediction:
                ax[2, i].imshow(
                    model.label_cmap[model.test_cluster_metrics.map_clusters(crf_outputs["cluster_preds"][img_num])])
            if compare_to_picie:
                ax[3, i].imshow(model.label_cmap[crf_outputs["picie_preds"][img_num]])
            ax[0, i].set_title(img_num, fontsize=16)

        ax[0, 0].set_ylabel("Image", fontsize=26)
        ax[1, 0].set_ylabel("Label", fontsize=26)
        if run_prediction:
            ax[2, 0].set_ylabel("STEGO (Ours)", fontsize=26)
        if compare_to_picie:
            ax[3, 0].set_ylabel("PiCIE (Baseline)", fontsize=26)

        remove_axes(ax)
        plt.tight_layout()
        plt.show()
        plt.clf()

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = model.test_cluster_metrics.histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(model.cfg.dataset_name)
    if model.cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [model.label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()
    plt.show()
    plt.clf()

    all_bars = torch.cat([
        model.test_cluster_metrics.histogram.sum(0).cpu(),
        model.test_cluster_metrics.histogram.sum(1).cpu()
    ], axis=0)
    ymin = max(all_bars.min() * .8, 1)
    ymax = all_bars.max() * 1.2

    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
    ax[0].bar(range(model.n_classes + model.cfg.extra_clusters),
              model.test_cluster_metrics.histogram.sum(0).cpu(),
              tick_label=names,
              color=colors)
    ax[0].set_ylim(ymin, ymax)
    ax[0].set_title("Label Frequency")
    ax[0].set_yscale('log')
    ax[0].tick_params(axis='x', labelrotation=90)

    ax[1].bar(range(model.n_classes + model.cfg.extra_clusters),
              model.test_cluster_metrics.histogram.sum(1).cpu(),
              tick_label=names,
              color=colors)
    ax[1].set_ylim(ymin, ymax)
    ax[1].set_title("Cluster Frequency")
    ax[1].set_yscale('log')
    ax[1].tick_params(axis='x', labelrotation=90)

    plt.tight_layout()
    plt.show()
    plt.clf()


if __name__ == "__main__":
    prep_args()
    my_app()
