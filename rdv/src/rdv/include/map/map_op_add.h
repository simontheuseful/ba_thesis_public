FORWARD {
    forward(parameters.map_a, _input, _output);
    float temp[OUTPUT_DIM];
    forward(parameters.map_b, _input, temp);
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] += temp[i];
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
backward(parameters.map_a, _input, _output_grad, _input_grad);
backward(parameters.map_b, _input, _output_grad, _input_grad);
#else
backward(parameters.map_a, _input, _output_grad);
backward(parameters.map_b, _input, _output_grad);
#endif
}