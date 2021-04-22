from hyperparameters import *
import random
import datetime


train_input_ind = [i for i in range(len(data["train"]["input"]))]


def now_to_string(format):
    now = datetime.datetime.now()
    string = now.strftime(format)
    return string


def elapsed(now):
    then = datetime.datetime.now()
    took = then - now
    mm, ss = divmod(took.seconds, 60)
    hh, mm = divmod(mm, 60)
    return "%dh %dm %ds" % (hh, mm, ss)


def get_batch(ind, to_use):
    batch = {"input": [], "label": []}

    for i in ind:
        batch["input"].append(data[to_use]["input"][i])
        batch["label"].append(data[to_use]["label"][i])

    batch["input"] = torch.Tensor(batch["input"]).to(device)
    batch["label"] = torch.LongTensor(batch["label"]).to(device)

    return batch


def get_random_train_batch():
    rand_ind = random.sample(train_input_ind, batch_size)
    return get_batch(rand_ind, "train")


def train_batch(batch):
    optimizer.zero_grad()
    output = model(batch["input"])
    loss = criterion(output, batch["label"])
    loss.backward()
    optimizer.step()
    scheduler.step()

    return loss.item() / batch_size
