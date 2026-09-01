This is my first neural network.

How to use:

1.Import

    from StasesNeuralNet import Network

2.Create net

    Net = Network(
            2.1.first_input_len
                length of first layer and input vector (int)

            2.2.layers_and_their_len
                list of layers and their length (list)
                example: [10, 20, 15, 10, 5, 2, 1]

            2.3.you can also set the learnint rate but its already 0.0001
                learning_rate=0.0001
        )

2.Fill the network

    Net.fill_network()

4.Make your dataset

5.Train net

    Net.train(input_vector, target_vector)

6.When the network is done, to get answer, use forward()

    answer = Net.forward(input_vector)

This litle project have only RELU
Ignore README.py, I have no idea how to delete this
