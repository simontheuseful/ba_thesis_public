FORWARD {
    forward(parameters.map_a, _input, _output);
    float temp[OUTPUT_DIM];
    forward(parameters.map_b, _input, temp);
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] -= temp[i];
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_a, _input, _output_grad, _input_grad);
#else
    backward(parameters.map_a, _input, _output_grad);
#endif
    float neg_output_grad[OUTPUT_DIM];
    for (int i=0; i<OUTPUT_DIM; i++) neg_output_grad[i] = -_output_grad[i];
#ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_b, _input, neg_output_grad, _input_grad);
#else
    backward(parameters.map_b, _input, neg_output_grad);
#endif
}