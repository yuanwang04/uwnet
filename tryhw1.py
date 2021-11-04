from uwnet import *

def conv_net():
    # Around 1.1M Ops in total
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),  # 8 * 27 X 27 * 1024 => 221,184 OP
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1), # 16 * 72 X 72 * 256 => 294,912 OP
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),  # 32 * 144 X 144 * 64 => 294,912 OP
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),  # 64 * 288 X 288 * 16 => 294,912 OP
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),                 # 2560 OP
            make_activation_layer(SOFTMAX)]
    return make_net(l)


def fc_net():
    # Around 1M Ops in total
    l = [
        make_connected_layer(3072, 300),  # 922k OP
        make_activation_layer(RELU),
        make_connected_layer(300, 200),   # 60k OP
        make_activation_layer(RELU),
        make_connected_layer(200, 100),   # 10k OP
        make_activation_layer(SOFTMAX)
    ]
    return make_net(l)


print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 1000
rate = .01
momentum = .9
decay = .005

m = conv_net()
# m = fc_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.
# Your answer:
# convnet does much better than the fully connected network.
# I think this is because convnet's archtecture utilizes the operations
# much better than normal fully connected network.
# convnet make connections between layers of the neural network focuing
# on the near pixels which favors how images are structures.
# Therefore a lot of connections(operations) are saved from less significant pixels far away.
# Such saved operations enabled us to have a much bigger network to deal with complex tasks
# regarding computer vision.

