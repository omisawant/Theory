class NeuralNetwork():
    def __init__(self, layer_counts, activation_function):
        self.layer_counts = layer_counts
        self.activation_function = activation_function
        # initialize weights and biases
        self.weight_matrix = []
        self.bias_matrix = []
        previous_layer_count = self.layer_counts[0]
        for layer_count in self.layer_counts[1:]:
            next_layer_count = layer_count
            perceptron_vector = [1]*(previous_layer_count)
            self.weight_matrix.append([perceptron_vector for _ in range(next_layer_count)])
            self.bias_matrix.append([1]*next_layer_count)
            previous_layer_count = next_layer_count

    def perceptron(self, input_vector, weight_vector, bias, output_layer):
        summation = sum([i*w for i,w in zip(input_vector, weight_vector)]) + bias
        if output_layer:
            return summation
        return summation, self.activation_function(summation)
    
    def forward_propagation(self, input_vector):
        layer_input_vector = input_vector
        summation_matrix = []
        input_and_activation_matrix = [input_vector]
        computed_output_vector = []
        layer_index = 0
        for layer_count in self.layer_counts[1:-1]:
            summation_vector = []
            activation_vector = []
            for perceptron_index in range(layer_count):
                summation, activation_result = self.perceptron(layer_input_vector, self.weight_matrix[layer_index][perceptron_index], self.bias_matrix[layer_index][perceptron_index], False)
                summation_vector.append(summation)
                activation_vector.append(activation_result)
            summation_matrix.append(summation_vector)
            input_and_activation_matrix.append(activation_vector)
            layer_input_vector = summation_vector
            layer_index = layer_index + 1
        for perceptron_index in range(self.layer_counts[-1]):
            computed_output = self.perceptron(layer_input_vector, self.weight_matrix[layer_index][perceptron_index], self.bias_matrix[layer_index][perceptron_index], True)
            computed_output_vector.append(computed_output)
        return summation_matrix, input_and_activation_matrix, computed_output_vector
    
    def compute(self, input_vector):
        return self.forward_propagation(input_vector)[2]
    
    def train(self, input_vectors, expected_output_vectors, d_loss_func, d_activation_func, epochs, learning_rate=None):
        for _ in range(epochs):
            updated_weight_matrix = self.weight_matrix.copy()
            for input_vector, expected_output_vector in zip(input_vectors, expected_output_vectors):
                summation_matrix, input_and_activation_matrix, computed_output_vector = self.forward_propagation(input_vector)
                d_loss_vector = [d_loss_func(c,e) for c,e in zip(computed_output_vector, expected_output_vector)]
                forward_vector = d_loss_vector
                for layer_index in range(len(input_and_activation_matrix)-1,-1,-1):
                    updated_forward_vector = []
                    for perceptron_index in range(len(input_and_activation_matrix[layer_index])):
                        weight_forward_value_product_sum = 0
                        for forward_vector_index in range(len(forward_vector)):
                            d_cost_func_wrt_weight = input_and_activation_matrix[layer_index][perceptron_index] * forward_vector[forward_vector_index]
                            current_weight = self.weight_matrix[layer_index][forward_vector_index][perceptron_index]
                            updated_weight_matrix[layer_index][forward_vector_index][perceptron_index] = current_weight - (learning_rate*d_cost_func_wrt_weight)
                            weight_forward_value_product_sum = weight_forward_value_product_sum + (current_weight * forward_vector[forward_vector_index])
                        updated_forward_vector.append(d_activation_func(summation_matrix[layer_index][perceptron_index]) * weight_forward_value_product_sum)
                    forward_vector = updated_forward_vector
            self.weight_matrix = updated_weight_matrix


# nn1 = NeuralNetwork(3,[2,2],3, int)
nn1 = NeuralNetwork([3,2,2,3], int)
print(nn1.weight_matrix)
# print(nn1.bias_matrix)
# print(nn1.perceptron([1,2,3],[4,5,6],8,False))
sm, iam, ov = nn1.forward_propagation([1,2,3])
print(sm)
print(iam)
print(ov)
# print(nn1.compute([1,2,3]))
