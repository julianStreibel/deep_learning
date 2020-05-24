import numpy as np
from matplotlib import pyplot as plt
from model import Sequential
from layer import Dense
from dataset_reader import get_dataset
from optimizer import stochastic_gradient_descent


train_x, train_y, = get_dataset("DATA", "train")
test_x, test_y = get_dataset("DATA", "test")
dev_x, dev_y = get_dataset("DATA", "dev")

def build_model(parameters):
    model = Sequential(optimizer=stochastic_gradient_descent(learning_rate=parameters[1], decay=parameters[2], momentum=parameters[3]))
    for units_of_layer in parameters[0]:
        model.add(Dense(units_of_layer))
    model.add(Dense(1))
    return model

def stochastic_search(n, layers=[0, 10], units=[1, 100], learning_rate=[0.01, 1], decay=[0, 0.5], momentum=[0, 1], epochs=[1, 100], batch_size=[1, 256]):
    """
    stochastic search for hyperparameter
    n: number of searches
    layers: [low, high] of number of hidden layers
    units: [low, high] of number of units in hidden layer
    learning_rate: [low, high] of learning rate
    decay: [low, high] of weight decay in backprop
    momentum: [low, high] of momentum in backprop
    epochs: [low, high] training epochs
    batch_size: [low, high] number of samples in a mini batch
    """
    best_parameters = None
    best_train_acc = None
    for i in range(n):
        n_units = np.random.randint(units[0], high=units[1], size=np.random.randint(layers[0], high=layers[1]))
        lr = learning_rate[0] + np.random.random() * (learning_rate[1] - learning_rate[0])
        d = decay[0] + np.random.random() * (decay[1] - decay[0])
        e = np.random.randint(epochs[0], high=epochs[1])
        bs = np.random.randint(batch_size[0], high=batch_size[1])
        m = momentum[0] + np.random.random() * (momentum[1] - momentum[0])

        parameters = [n_units, lr, d, m, e, bs]

        model = build_model(parameters)
        print(f'\nStarting with model {i + 1}')
        print('params', parameters)
        model.fit(train_x, train_y, test_x, test_y, e, batch_size=bs)
        acc = model.binary_accuracy(test_x, test_y)
        print(f'trainings acc {round(acc * 100, 2)} %')

        if best_train_acc == None or best_train_acc < acc:
            best_train_acc = acc
            best_parameters = parameters

    print('–' * 50)
    best_model = build_model(best_parameters)
    train_history, test_history, loss_history = best_model.fit(train_x, train_y, test_x, test_y, e, batch_size=bs, history=True)
    print('Best model on test set')
    print('layers:', best_parameters[0])
    print('learning rate:', best_parameters[1])
    print('decay:', best_parameters[2])
    print('momentum:', best_parameters[3])
    print('epochs:', best_parameters[4])
    print('batch size:', best_parameters[5])
    print('Training accuracy:', best_model.binary_accuracy(train_x, train_y))
    print('Test accuracy:', best_model.binary_accuracy(test_x, test_y))
    print('Dev accuracy:', best_model.binary_accuracy(dev_x, dev_y))
    print('–' * 50)

    x = range(len(train_history))
    _, axs = plt.subplots(2)
    axs[0].set_title("Accurecy")
    axs[0].plot(x, train_history, label='acc_train')
    axs[0].plot(x, test_history, label='acc_test')
    axs[1].set_title("Loss")
    axs[1].plot(x, loss_history, label='loss_train')
    plt.show()



stochastic_search(100, layers=[2, 7], units=[2, 75], learning_rate=[0.01, 0.45], decay=[0, 0.05], epochs=[50, 150], batch_size=[1, 256])