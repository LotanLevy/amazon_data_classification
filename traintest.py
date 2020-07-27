

import matplotlib.pyplot as plt
import os


def train(epochs, batch_size, trainer, validator, train_dataloader, val_dataloader, print_freq, output_path, model):
    max_iteration = epochs * batch_size
    trainstep = trainer.get_step()
    # valstep = validator.get_step()
    # logger = TrainLogger(trainer, validator, output_path)
    for i in range(max_iteration):
        batch_x, batch_y = train_dataloader.read_batch(batch_size)
        trainstep(batch_x, batch_y)
        # if i % print_freq == 0:
        #     batch_x, batch_y = val_dataloader.read_batch(batch_size)
        #     valstep(batch_x, batch_y)
        #     logger.update(i)
        #
        # if i % epochs == 0:
        #     model.save_model(int(i/epochs), output_path)

def check_accuracy(validator, test_dataloader, test_number):
    batch_x, batch_y, paths, labels = test_dataloader.read_batch_with_details(test_number)
    valstep = validator.get_step()
    valstep(batch_x, batch_y)






def plot_dict(dict, x_key, output_path):
    for key in dict:
        if key != x_key:
            f = plt.figure()
            plt.plot(dict[x_key], dict[key])
            plt.title(key)
            plt.savefig(os.path.join(output_path, key))
            plt.close(f)
    plt.close("all")


class TrainLogger:
    def __init__(self, trainer, validator, output_path):
        self.logs = {"iteration": [], "train_D_loss": [], "train_C_loss": [], "val_D_loss": [], "val_C_loss": []}
        self.trainer = trainer
        self.validator = validator
        self.output_path = output_path

    def update(self, iteration):
        self.logs["iteration"].append(iteration)
        self.logs["train_D_loss"].append(float(self.trainer.D_loss_mean.result()))
        self.logs["train_C_loss"].append(float(self.trainer.C_loss_mean.result()))
        self.logs["val_D_loss"].append(float(self.validator.D_loss_mean.result()))
        self.logs["val_C_loss"].append(float(self.validator.C_loss_mean.result()))

    def __del__(self):
        plot_dict(self.logs, "iteration", self.output_path)








