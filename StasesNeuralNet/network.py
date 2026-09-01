import random

class Neuron:
    def __init__(self, weights, bias=0):
        self.weights=weights
        self.bias =bias
        self.error =0
        self.weight_delta =[]
        self.bias_delta =0

    def forward(self, input_vector):
        self.last_input=input_vector
        answer=0
        for i in range(len(input_vector)):
            answer += input_vector[i] *self.weights[i]
        answer+=self.bias
        self.z=answer
        answer=self.activation_func(answer)
        self.output=answer
        return answer

    def activation_func(self, answer):
        return max(0, answer)

    def count_delta(self, learning_rate):
        self.weight_delta =[]
        for w in range( len( self.weights)):
            self.weight_delta.append( self.last_input[ w] *self.error *learning_rate)

        self.bias_delta =self.error *learning_rate

    def apply_delta(self):
        for w in range( len( self.weights)):
            self.weights[w] +=self.weight_delta[w]
        self.bias +=self.bias_delta

        self.weight_delta =[]
        self.bias_delta =0
        self.error =0

    def show_info(self):
        print('weights:',self.weights)
        print('bias:',self.bias)
        print('error:',self.error)
        print('weight delta:',self.weight_delta)
        print('bias delta:',self.bias_delta,'\n')

    def activation_derivative(self):
        if self.z <=0: return 0
        else: return 1

class Network:
    def __init__(self, first_input_len, layers_and_their_len, learning_rate=0.0001):
        self.layers =[]
        self.input_len=first_input_len
        self.LAndTLen=layers_and_their_len
        self.learning_rate=learning_rate

    def create_neuron(self, weights_count, weights=None):
        if not weights:
            weights=[]
            std = (2 / weights_count) ** 0.5

            for i in range(weights_count):
                weights.append(random.gauss(0, std))
        return Neuron(weights)

    def fill_network(self):
        for lyr in range(len(self.LAndTLen)):
            layer=[]
            if lyr ==0:
                weights_count=self.input_len
            else:
                weights_count=self.LAndTLen[lyr-1]
            for n in range(self.LAndTLen[lyr]):
                layer.append( self.create_neuron(weights_count))
            self.layers.append(layer)

    def show_info(self):
        for l in range(len(self.layers)):  # перевірка
            print('\n=#$%#*=~- ' + str(l + 1) + ' layer -~=*#%$#=')
            for n in range(len(self.layers[l])):
                self.layers[l][n].show_info()

    def layer_forward(self, layer, input_vector):
        new_vector=[]
        for neuron in self.layers[ layer]:
            new_vector.append( neuron.forward( input_vector))
        return new_vector

    def forward(self, input_vector):
        result=[]
        for lyr in range(len(self.layers)):
            result =self.layer_forward(lyr,input_vector)
            input_vector=result
        self.output=result
        return result

    def train(self, input_vector, target):
        self.forward(input_vector)
        self.calculate_output_error(target)
        loss =self.calculate_loss(target)
        self.calculate_hidden_errors()
        self.count_deltas()
        self.apply_deltas()
        return loss

    def calculate_output_error(self, target):
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer):
            error = (target[i] - neuron.output) * neuron.activation_derivative()
            neuron.error = error

    def calculate_hidden_errors(self):
        for layer_index in range((len(self.layers) -2), -1, -1):
            current_layer =self.layers[layer_index]
            next_layer =self.layers[layer_index +1]
            for i, neuron in enumerate(current_layer):
                sum_error=0
                for next_neuron in next_layer:
                    sum_error +=next_neuron.error *next_neuron.weights[i]

                sum_error *=neuron.activation_derivative()
                neuron.error =sum_error

    def count_deltas(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.count_delta(self.learning_rate)

    def apply_deltas(self):
        for layer in self.layers:
            for neuron in layer:
                neuron.apply_delta()

    def calculate_loss(self,target):
        loss =0
        last_layer = self.layers[-1]
        for i, neuron in enumerate(last_layer):
            loss +=(target[i] -neuron.output) **2
        loss /= len(last_layer)
        return loss