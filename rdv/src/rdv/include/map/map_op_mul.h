/*
This map operator multiplies the outputs of two input maps element-wise.
*/
FORWARD {
    forward(parameters.map_a, _input, _output);
    float temp[OUTPUT_DIM];
    forward(parameters.map_b, _input, temp);
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] *= temp[i];
}

BACKWARD {
    #ifdef STOCHASTIC
    uvec4 seed = random_seed();
    #endif
    float _temp_a[OUTPUT_DIM];
    forward(parameters.map_a, _input, _temp_a);
    float _temp_b[OUTPUT_DIM];
    forward(parameters.map_b, _input, _temp_b);
    #ifdef STOCHASTIC
    seed = random_seed(seed);  // restore seed before map_a
    #endif
    float dL_darg[OUTPUT_DIM];
    for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * _temp_b[i];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_a, _input, _temp_a, dL_darg, _input_grad);
    #else
    backward(parameters.map_a, _input, _temp_a, dL_darg);
    #endif
    for (int i=0; i<OUTPUT_DIM; i++) dL_darg[i] = _output_grad[i] * _temp_a[i];
    #ifdef INPUT_REQUIRES_GRAD
    backward(parameters.map_b, _input, _temp_b, dL_darg, _input_grad);
    #else
    backward(parameters.map_b, _input, _temp_b, dL_darg);
    #endif
    #ifdef STOCHASTIC
    random_seed(seed);  // restore seed after map_b
    #endif
}