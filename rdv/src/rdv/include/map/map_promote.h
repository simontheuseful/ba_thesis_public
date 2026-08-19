FORWARD {
    float r[1];
    forward(parameters.map, _input, r);
    for (int i=0; i<OUTPUT_DIM; i++) _output[i] = r[0];
}

BACKWARD {
    float dL_dr[1];
    dL_dr[0] = 0.0;
    for (int i=0; i<OUTPUT_DIM; i++)
        dL_dr[0] += _output_grad[i];
#ifndef INPUT_REQUIRES_GRAD
    #ifndef BW_USES_OUTPUT
    backward(parameters.map, _input, dL_dr);
    #else
    backward(parameters.map, _input, float[1](_output[0]), dL_dr);
    #endif
#else
    #ifndef BW_USES_OUTPUT
    backward(parameters.map, _input, dL_dr, _input_grad);
    #else
    backward(parameters.map, _input, float[1](_output[0]), dL_dr, _input_grad);
    #endif
#endif
}