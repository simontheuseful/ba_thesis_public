FORWARD {
    forward(parameters.map_a, _input, _output);
    float _temp[OUTPUT_DIM];
    forward(parameters.map_b, _input, _temp);
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] /= _temp[i];
}

BACKWARD {
#ifdef STOCHASTIC
    uvec4 seed = random_seed();
#endif
    float a[OUTPUT_DIM];
    forward(parameters.map_a, _input, a);
    float b[OUTPUT_DIM];
    forward(parameters.map_b, _input, b);
#ifdef STOCHASTIC
    seed = random_seed(seed);  // restore seed before map_a
#endif
    float dL_darg[OUTPUT_DIM];
    for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] / b[i];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_a, _input, a, dL_darg, _input_grad);
    #else
    backward(parameters.map_a, _input, a, dL_darg);
    #endif
    for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] *= - a[i] / b[i];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_b, _input, b, dL_darg, _input_grad);
    #else
    backward(parameters.map_b, _input, b, dL_darg);
    #endif
#ifdef STOCHASTIC
    random_seed(seed);  // restore seed after map_b
#endif
}