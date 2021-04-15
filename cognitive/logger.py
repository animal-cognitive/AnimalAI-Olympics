import csv
import datetime
import shutil
import subprocess

# import paramiko
import torch
from tensorboardX import SummaryWriter
import os

from cognitive.global_params import *
"""
MIT License
Jason Hu
"""

class Logger:
    """
    Use the logger to output csv of training statistics, save/load models
    The logger creates a unique directory so you can run experiments in parallel.
    """

    def __init__(self, set_name="GAN", version="v0", disabled=False, writer=True):
        """

        :param set_name:
        :param version:
        :param disabled: disable if rank is not 0 or -1
        :param writer:
        """
        self.set = set_name
        self.version = version
        self.disabled = disabled

        # the overall saves directory is paralell to the project directory
        # this is so that my IDE does not reindex the files

        # take a look at the file structure defined here
        self.save_dir = save_dir
        if not self.save_dir.exists() and not disabled:
            self.save_dir.mkdir()

        self.set_dir = self.save_dir / self.set
        if not self.set_dir.exists() and not disabled:
            self.set_dir.mkdir()

        self.version_dir = self.set_dir / self.version
        if not self.version_dir.exists() and not disabled:
            self.version_dir.mkdir()

        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
        self.exp_dir = self.version_dir / timestamp
        if not self.exp_dir.exists() and not disabled:
            self.exp_dir.mkdir()

        self.model_check_points_dir = self.exp_dir / "checkpoints"
        if not self.model_check_points_dir.exists() and not disabled:
            self.model_check_points_dir.mkdir()

        self.log_dir = self.exp_dir / "log"
        if not self.log_dir.exists() and not disabled:
            self.log_dir.mkdir()

        self.csv_dir = self.exp_dir / "csv"
        if not self.csv_dir.exists() and not disabled:
            self.csv_dir.mkdir()

        self.log_file = self.new_log_file_path()
        self.csv_field_names = {}
        if writer:
            self.make_writer()

    @property
    def path(self):
        return self.exp_dir

    def make_writer(self):
        self.writer = SummaryWriter(self.path / "summary")
        return self.writer

    # def auto_write(self, prepend, csv_name=None, replace=True, **column_values):
    #     self.auto_log(prepend, csv_name, replace, **column_values)

    def auto_log(self, prepend, csv_name=None, replace=True, tensorboard=True, **column_values):
        """
        Overwrites the log
        Unlikely though, since the directories are timestamped
        :param prepend:
        :param csv_name:
        :param replace:
        :param: the column, values
        :return:
        """

        if not self.disabled:
            if csv_name is None:
                csv_name = self.auto_csv_name(prepend)

            columns, values = column_values.keys(), column_values.values()
            column_value_dict = column_values

            if csv_name not in self.csv_field_names:
                self.new_csv(csv_name, columns, replace=replace)

            self.csv_write_row(csv_name, column_value_dict)

        self.log_print(f"{'======' + prepend + '======':^60}".upper())
        # headers=f"{prepend}||"
        # values=f"{'':{len(prepend)}}||"
        headers = "|"
        values = "|"
        for i, (col, val) in enumerate(column_values.items()):
            wid = len(col) if len(col) > 8 else 8

            if isinstance(val, int):
                val_str = f"{val:{wid}}|"
            elif isinstance(val, float):
                val_str = f"{val:{wid}.4f}|"
            elif isinstance(val, torch.Tensor):
                val = val.item()
                val_str = f"{val:{wid}.4f}|"
            else:
                try:
                    val = float(val)
                    val_str = f"{val:{wid}.4f}|"
                except TypeError:
                    if isinstance(val, AverageMeter):
                        val = val.avg
                        val_str = f"{val:{wid}}|"
                    else:
                        raise NotImplementedError(str(val.__class__) + " cannot be logged")
            if tensorboard:
                if col.lower() not in ("epoch", "iter", "iteration"):
                    if "epoch" in column_values:
                        self.writer.add_scalar(prepend + '/' + col, val, column_values["epoch"])
                    else:
                        self.writer.add_scalar(prepend + '/' + col, val)

            values += val_str
            headers += f"{col:>{wid}}|"

            if (i + 1) % 8 == 0:
                self.log_print(headers)
                self.log_print(values)
                headers = "|"
                values = "|"
                # headers = f"{prepend}||"
                # values = f"{'':{len(prepend)}}||"

        if len(column_values) % 8 != 0:
            self.log_print(headers)
            self.log_print(values)

    def load_pickle(self, time_stamp, starting_epoch=None, starting_iteration=None):
        """
        In favor of load through pickle rather than state_dict, because some torch objects we want to save
        do not have state dict.
        Do not load individual models. All optimizers/models/auxiliary objects need to be saved/loaded at
        the same time.
        :param starting_epoch: You can specify the epoch/iteration of model you want to load.
        :param starting_iteration:
        :return:
        """

        highest_epoch = 0
        highest_iter = 0

        time_stamp_dir = self.version_dir / time_stamp / "checkpoints"

        for child in time_stamp_dir.iterdir():
            if child.name.split("_")[0] == self.version:
                try:
                    epoch = child.name.split("_")[1]
                    iteration = child.name.split("_")[2].split('.')[0]
                except IndexError:
                    print(str(child))
                    raise
                iteration = int(iteration)
                epoch = int(epoch)
                # some files are open but not written to yet.
                if child.stat().st_size > 128:
                    if epoch > highest_epoch or (iteration > highest_iter and epoch == highest_epoch):
                        highest_epoch = epoch
                        highest_iter = iteration
        if highest_epoch == 0 and highest_iter == 0:
            print("nothing to load")
            return None

        if starting_epoch is None and starting_iteration is None:
            # load the highest epoch, iteration
            pickle_file = time_stamp_dir / (
                    self.version + "_" + str(highest_epoch) + "_" + str(highest_iter) + ".pkl")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                payload = torch.load(pickle_file)
            print('Loaded model at epoch ', highest_epoch, 'iteration', highest_iter)
        else:
            pickle_file = time_stamp_dir / (
                    self.version + "_" + str(starting_epoch) + "_" + str(starting_iteration) + ".pkl")
            if not pickle_file.exists():
                raise FileNotFoundError("The model checkpoint does not exist")
            print("loading model at", pickle_file)
            with pickle_file.open('rb') as pickle_file:
                payload = torch.load(pickle_file)
            print('Loaded model at epoch ', starting_epoch, 'iteration', starting_iteration)

        return payload, highest_epoch, highest_iter

    def save_pickle(self, payload, epoch, iteration=0, is_best=False):
        if self.disabled:
            print("Not saved, logger disabled")
        else:
            epoch = int(epoch)
            pickle_file = self.model_check_points_dir / (
                    self.version + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
            with pickle_file.open('wb') as fhand:
                torch.save(payload, fhand)
            if is_best:
                pickle_file = self.model_check_points_dir / (
                        "best_" + self.version + "_" + str(epoch) + "_" + str(iteration) + ".pkl")
                with pickle_file.open('wb') as fhand:
                    torch.save(payload, fhand)

            print(f"Saved model set {self.set} version {self.version} at {pickle_file}")

    def new_log_file_path(self):
        """
        Does not make the file.
        :return:
        """
        log_file = self.log_dir / "log.txt"
        self.log_file = log_file
        return log_file

    def log_print(self, string, print_it=True):
        if not self.disabled:
            string = str(string)
            if self.log_file is not None and self.log_file is not None:
                with open(self.log_file, 'a') as handle:
                    handle.write(string + '\n')
        if print_it:
            print(string)

    def new_csv(self, csv_name, field_names, replace=False):
        """
        :param csv_name: the name of the csv
        :param field_names:
        :param replace: replace the csv file
        :return:
        """
        # safety checks
        csv_path = self.csv_dir / csv_name
        if not csv_path.suffix == ".csv":
            raise ValueError("use .csv file names")
        if csv_path.exists():
            if not replace:
                raise FileExistsError("the csv file already exists")
            else:
                os.remove(csv_path)

        with open(csv_path, 'w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writeheader()

        self.csv_field_names[csv_name] = field_names

    def csv_write_row(self, csv_name, column_values_dict):
        """
        :param csv_name:
        :param column_values_dict: a dictionary column : value
        :return:
        """

        csv_path = self.csv_dir / csv_name
        if not csv_path.suffix == ".csv":
            raise ValueError("use .csv file names")
        if not csv_path.exists():
            raise FileNotFoundError("The csv you are writing to is not registered")

        field_names = self.csv_field_names[csv_name]
        for key in column_values_dict.keys():
            assert key in field_names
        for fn in field_names:
            assert fn in column_values_dict

        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=field_names)
            writer.writerow(column_values_dict)

    def auto_csv_name(self, prepend):
        # make csv_name automatically
        words = prepend.split()
        csv_name = ""
        for word in words:
            csv_name += word[:4].lower()
        return csv_name + ".csv"

    def close(self):
        self.writer.close()


def print_file(filename):
    with open(filename, "r") as handle:
        for line in handle:
            print(line)


class TensorBoardServer:
    def __init__(self):
        self.selected_dir = save_dir.parent / "selected_tensorboard"
        self.tar_name = "tensorboard.tar.gz"
        self.ssh = None

    def select_summaries(self):
        exclude = ["covready", "alpha", "FixedNLP", "nullagent"]

        shutil.rmtree(self.selected_dir)
        self.selected_dir.mkdir(exist_ok=True)
        fs = list(save_dir.rglob("*"))
        for f in fs:
            if "tfevents" in f.name:
                if f.stat().st_size > 10000:
                    if all(i not in str(f.absolute()) for i in exclude):
                        target_path = self.selected_dir / f.relative_to(save_dir)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(f, target_path)
                    else:
                        target_path = self.selected_dir / "earlier_runs" / f.relative_to(save_dir)
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(f, target_path)

        subprocess.run(["rm", self.tar_name], cwd=self.selected_dir.parent)
        subprocess.run(["tar", "-zcvf", self.tar_name, self.selected_dir.name], cwd=self.selected_dir.parent)

    def sftp_send_tar(self):
        sftp = self.ssh.open_sftp()
        sftp.put(self.selected_dir.parent / self.tar_name, f"/root/web/{self.tar_name}")

        # user = "root"
        # host = "philosophymachine.com"
        # remote_dir = "~"
        # local_file_path = self.tar_name
        # subprocess.run(f"sftp {user}@{host}:{remote_dir} <<< $\'put {local_file_path}\'",
        #                cwd=self.temp_dir, shell=True)

    def open_ssh(self):
        ss = paramiko.SSHClient()
        ss.load_system_host_keys()
        ss.connect("philosophymachine.com", username="root", look_for_keys=True)
        self.ssh = ss

    def run(self, cmd):
        return self.ssh.exec_command(cmd)

    def untar_and_serve(self):
        self.run(f"cd /root/web && ./refresh.sh")

    def close(self):
        self.ssh.close()


class MultiAverageMeter:
    def __init__(self):
        self.dict = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.dict:
                self.dict[key] = AverageMeter()
            try:
                value = value.item()
            except (ValueError, AttributeError):
                pass
            am = self.dict[key]
            am.update(value)

    def get(self):
        return {key: value.avg for key, value in self.dict.items()}


class AverageMeter:
    """
    Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.M2 = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        delta = val - self.avg
        self.avg = self.sum / self.count
        delta2 = val - self.avg
        self.M2 += delta * delta2

    def __float__(self):
        return self.avg

    def avg(self):
        return self.avg

    @property
    def var(self):
        return self.M2 / (self.count - 1)


def fast_simple_refresh():
    tbs = TensorBoardServer()
    tbs.select_summaries()
    tbs.open_ssh()
    tbs.sftp_send_tar()
    # i, o, e = tbs.run("cd /root/web && ls -l")
    # print(o.read().decode("utf-8"))
    tbs.untar_and_serve()
    tbs.close()


if __name__ == '__main__':
    fast_simple_refresh()
