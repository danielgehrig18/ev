import numpy as np

def convolution(input, weight):
    # weight c_out x c_in x kernel_size x kernel_size
    # input: c_in x height x width
    c_out, c_in, kernel_size, _ = weight.shape
    _, height, width = input.shape
    output = np.zeros((c_out, height, width))
    for i in range(height):
        for j in range(width):
            for u in range(kernel_size):
                for v in range(kernel_size):
                    y = i + u - kernel_size // 2
                    x = j + v - kernel_size // 2
                    if y < height and x < width and x >= 0 and y >= 0:
                        output[:,i,j] += weight[..., u, v] @ input[:, y, x]
    return output

def project_weight(weight):
    # weight c_out x c_in x kernel_size x kernel_size
    c_out, c_in, _, _ = weight.shape
    for c_o in range(c_out):
        for c_i in range(c_in):
            weight[c_out - c_o - 1, c_in - c_i - 1] = weight[c_o, c_i]
    return weight

def lift_weight(weight):
    return np.concatenate((weight, weight[::-1,::-1]))


if __name__ == '__main__':
    kernel_size = 3
    c_out = 16
    c_in = 5
    height = 180
    width = 240

    weight = np.random.rand(c_out, c_in, kernel_size, kernel_size)
    #weight = project_weight(weight)
    weight = lift_weight(weight)

    input = np.random.rand(c_in, height, width)
    input_tilde = - input[::-1]

    output = convolution(input, weight)
    output_tilde_test = -output[::-1]
    output_tilde = convolution(input_tilde, weight)
    print(np.abs(output_tilde_test - output_tilde).max())


