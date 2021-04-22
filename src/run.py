from utils import *


def train():
    print("------------------------------------------")
    print(" Training started at %s." % now_to_string("%F %T"))
    print("------------------------------------------")

    now = datetime.datetime.now()
    model.train()
    avg_loss = 0
    for iter in range(1, n_iter + 1):
        batch = get_random_train_batch()
        loss = train_batch(batch)
        avg_loss += loss

        if iter % print_freq == 0:
            sys.stdout.write("%3g%%: %.6f " % ((iter / n_iter) * 100, avg_loss / print_freq))
            sys.stdout.write("(" + elapsed(now) + ')\n')
            avg_loss = 0
    print()


def evaluate(to_use):
    print("--------------------------------------------")
    print(" Validation started at %s." % now_to_string("%F %T"))
    print("--------------------------------------------")

    now = datetime.datetime.now()
    model.eval()
    correct = 0
    iter = 0

    while iter < len(data[to_use]["input"]):
        try:
            assert iter + batch_size < len(data[to_use]["input"])
            ind = [i for i in range(iter, iter + batch_size)]
        except AssertionError:
            ind = [i for i in range(iter, len(data[to_use]["input"]))]

        batch = get_batch(ind, to_use)
        output = model(batch["input"])

        for b in range(len(ind)):
            if torch.argmax(output[b]) == batch["label"][b]:
                correct += 1

        iter += batch_size

    sys.stdout.write("Elapsed " + elapsed(now) + '\n')
    return correct / len(data[to_use]["input"])