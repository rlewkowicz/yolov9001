import os
import warnings
import pkg_resources as pkg
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.general import LOGGER, colorstr, cv2
from utils.loggers.clearml.clearml_utils import ClearmlLogger
from utils.loggers.wandb.wandb_utils import WandbLogger
from utils.plots import plot_images, plot_labels, plot_results
from utils.torch_utils import de_parallel

LOGGERS = ("csv", "tb", "wandb", "clearml", "comet")
RANK = int(os.getenv("RANK", -1))
try:
    import wandb

    assert hasattr(wandb, "__version__")
    if pkg.parse_version(wandb.__version__) >= pkg.parse_version("0.12.2") and RANK in {
        0,
        -1,
    }:
        try:
            wandb_login_success = wandb.login(timeout=30)
        except wandb.errors.UsageError:
            wandb_login_success = False
        if not wandb_login_success:
            wandb = None
except (ImportError, AssertionError):
    wandb = None
try:
    import clearml

    assert hasattr(clearml, "__version__")
except (ImportError, AssertionError):
    clearml = None
try:
    if RANK not in [0, -1]:
        comet_ml = None
    else:
        import comet_ml

        assert hasattr(comet_ml, "__version__")
        from utils.loggers.comet import CometLogger
except (ModuleNotFoundError, ImportError, AssertionError):
    comet_ml = None

class Loggers:
    def __init__(
        self,
        save_dir=None,
        weights=None,
        opt=None,
        hyp=None,
        logger=None,
        include=LOGGERS,
    ):
        self.save_dir = save_dir
        self.weights = weights
        self.opt = opt
        self.hyp = hyp
        self.plots = not opt.noplots
        self.logger = logger
        self.include = include

        self.keys = [
            "train/box_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "metrics/precision",
            "metrics/recall",
            "metrics/mAP_0.5",
            "metrics/mAP_0.5:0.95",
            "val/box_loss",
            "val/cls_loss",
            "val/dfl_loss",
        ]

        self.best_keys = [
            "best/epoch",
            "best/precision",
            "best/recall",
            "best/mAP_0.5",
            "best/mAP_0.5:0.95",
        ]
        for k in LOGGERS:
            setattr(self, k, None)
        self.csv = True

        if not clearml:
            prefix = colorstr("ClearML: ")
            s = f"{prefix}run 'pip install clearml' to automatically track, visualize and remotely train YOLO 🚀 in ClearML"
            self.logger.info(s)

        if not comet_ml:
            prefix = colorstr("Comet: ")
            s = f"{prefix}run 'pip install comet_ml' to automatically track and visualize YOLO 🚀 runs in Comet"
            self.logger.info(s)

        s = self.save_dir

        if "tb" in self.include and (not self.opt.evolve):
            prefix = colorstr("TensorBoard: ")
            self.logger.info(
                f"{prefix}Start with 'tensorboard --logdir {s.parent}', view at http://localhost:6006/"
            )
            self.tb = SummaryWriter(str(s))

        if wandb and "wandb" in self.include:
            wandb_artifact_resume = isinstance(self.opt.resume, str
                                              ) and self.opt.resume.startswith("wandb-artifact://")
            run_id = (
                torch.load(self.weights).get("wandb_id") if self.opt.resume and
                (not wandb_artifact_resume) else None
            )
            self.opt.hyp = self.hyp
            self.wandb = WandbLogger(self.opt, run_id)
        else:
            self.wandb = None

        if clearml and "clearml" in self.include:
            self.clearml = ClearmlLogger(self.opt, self.hyp)
        else:
            self.clearml = None

        if comet_ml and "comet" in self.include:
            if isinstance(self.opt.resume, str) and self.opt.resume.startswith("comet://"):
                run_id = self.opt.resume.split("/")[-1]
                self.comet_logger = CometLogger(self.opt, self.hyp, run_id=run_id)
            else:
                self.comet_logger = CometLogger(self.opt, self.hyp)
        else:
            self.comet_logger = None

    @property
    def remote_dataset(self):
        data_dict = None
        if self.clearml:
            data_dict = self.clearml.data_dict
        if self.wandb:
            data_dict = self.wandb.data_dict
        if self.comet_logger:
            data_dict = self.comet_logger.data_dict
        return data_dict

    def on_train_start(self):
        if self.comet_logger:
            self.comet_logger.on_train_start()

    def on_pretrain_routine_start(self):
        if self.comet_logger:
            self.comet_logger.on_pretrain_routine_start()

    def on_pretrain_routine_end(self, labels, names):
        if self.plots:
            plot_labels(labels, names, self.save_dir)
            paths = self.save_dir.glob("*labels*.jpg")
            if self.wandb:
                self.wandb.log({"Labels": [wandb.Image(str(x), caption=x.name) for x in paths]})
            if self.comet_logger:
                self.comet_logger.on_pretrain_routine_end(paths)

    def on_train_batch_end(self, model, ni, imgs, targets, paths, vals):
        log_dict = dict(zip(self.keys[0:3], vals))
        if self.plots:
            if ni < 3:
                f = self.save_dir / f"train_batch{ni}.jpg"
                plot_images(imgs, targets, paths, f)
                if ni == 0 and self.tb and (not self.opt.sync_bn):
                    log_tensorboard_graph(self.tb, model, imgsz=(self.opt.imgsz, self.opt.imgsz))
            if ni == 10 and (self.wandb or self.clearml):
                files = sorted(self.save_dir.glob("train*.jpg"))
                if self.wandb:
                    self.wandb.log({
                        "Mosaics": [
                            wandb.Image(str(f), caption=f.name) for f in files if f.exists()
                        ]
                    })
                if self.clearml:
                    self.clearml.log_debug_samples(files, title="Mosaics")
        if self.comet_logger:
            self.comet_logger.on_train_batch_end(log_dict, step=ni)

    def on_train_epoch_end(self, epoch):
        if self.wandb:
            self.wandb.current_epoch = epoch + 1
        if self.comet_logger:
            self.comet_logger.on_train_epoch_end(epoch)

    def on_val_start(self):
        if self.comet_logger:
            self.comet_logger.on_val_start()

    def on_val_batch_end(self, batch_i, im, targets, paths, shapes, out):
        if self.comet_logger:
            self.comet_logger.on_val_batch_end(batch_i, im, targets, paths, shapes, out)

    def on_val_end(self, nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix):
        if self.wandb or self.clearml:
            files = sorted(self.save_dir.glob("val*.jpg"))
            if self.wandb:
                self.wandb.log({"Validation": [wandb.Image(str(f), caption=f.name) for f in files]})
            if self.clearml:
                self.clearml.log_debug_samples(files, title="Validation")
        if self.comet_logger:
            self.comet_logger.on_val_end(nt, tp, fp, p, r, f1, ap, ap50, ap_class, confusion_matrix)

    def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
        num_vals = len(vals)
        num_keys = len(self.keys)
        if num_vals > num_keys:
            num_lrs = num_vals - num_keys
            self.keys.extend([f'x/lr{i}' for i in range(num_lrs)])

        x = dict(zip(self.keys, vals))

        if self.csv:
            file = self.save_dir / "results.csv"
            n = len(x) + 1
            s = (
                "" if file.exists() else
                ("%20s," * n % tuple(["epoch"] + self.keys)).rstrip(",") + "\n"
            )
            with open(file, "a") as f:
                f.write(s + ("%20.5g," * n % tuple([epoch] + vals)).rstrip(",") + "\n")

        if self.tb:
            for k, v in x.items():
                self.tb.add_scalar(k, v, epoch)
        elif self.clearml:
            for k, v in x.items():
                (title, series) = k.split("/")
                self.clearml.task.get_logger().report_scalar(title, series, v, epoch)

        if self.wandb:
            if best_fitness == fi:
                best_results = [epoch] + vals[3:7]
                for i, name in enumerate(self.best_keys):
                    self.wandb.wandb_run.summary[name] = best_results[i]
            self.wandb.log(x)
            self.wandb.end_epoch(best_result=best_fitness == fi)

        if self.clearml:
            self.clearml.current_epoch_logged_images = set()
            self.clearml.current_epoch += 1

        if self.comet_logger:
            self.comet_logger.on_fit_epoch_end(x, epoch=epoch)

    def on_model_save(self, last, epoch, final_epoch, best_fitness, fi):
        if ((epoch + 1) % self.opt.save_period == 0 and (not final_epoch) and
            (self.opt.save_period != -1)):
            if self.wandb:
                self.wandb.log_model(
                    last.parent, self.opt, epoch, fi, best_model=best_fitness == fi
                )
            if self.clearml:
                self.clearml.task.update_output_model(
                    model_path=str(last),
                    model_name="Latest Model",
                    auto_delete_file=False,
                )
        if self.comet_logger:
            self.comet_logger.on_model_save(last, epoch, final_epoch, best_fitness, fi)

    def on_train_end(self, last, best, epoch, results):
        if self.plots:
            plot_results(file=self.save_dir / "results.csv")
        files = [
            "results.png",
            "confusion_matrix.png",
            *(f"{x}_curve.png" for x in ("F1", "PR", "P", "R")),
        ]
        files = [self.save_dir / f for f in files if (self.save_dir / f).exists()]
        self.logger.info(f"Results saved to {colorstr('bold', self.save_dir)}")

        if self.tb and (not self.clearml):
            for f in files:
                self.tb.add_image(f.stem, cv2.imread(str(f))[..., ::-1], epoch, dataformats="HWC")

        if self.wandb:
            self.wandb.log(dict(zip(self.keys[3:10], results)))
            self.wandb.log({"Results": [wandb.Image(str(f), caption=f.name) for f in files]})
            if not self.opt.evolve:
                wandb.log_artifact(
                    str(best if best.exists() else last),
                    type="model",
                    name=f"run_{self.wandb.wandb_run.id}_model",
                    aliases=["latest", "best", "stripped"],
                )
            self.wandb.finish_run()

        if self.clearml and (not self.opt.evolve):
            self.clearml.task.update_output_model(
                model_path=str(best if best.exists() else last),
                name="Best Model",
                auto_delete_file=False,
            )

        if self.comet_logger:
            final_results = dict(zip(self.keys[3:10], results))
            self.comet_logger.on_train_end(files, self.save_dir, last, best, epoch, final_results)

    def on_params_update(self, params: dict):
        if self.wandb:
            self.wandb.wandb_run.config.update(params, allow_val_change=True)
        if self.comet_logger:
            self.comet_logger.on_params_update(params)

def log_tensorboard_graph(tb, model, imgsz=(640, 640)):
    try:
        p = next(model.parameters())
        imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tb.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ TensorBoard graph visualization failure {e}")
