from run import train, evaluate


if __name__ == "__main__":
    train()
    accuracy = evaluate("valid")
    print("Accuracy: %.2f%%" % (accuracy * 100))