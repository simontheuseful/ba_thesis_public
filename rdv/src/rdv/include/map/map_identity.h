FORWARD {
    _output = _input;
}

BACKWARD {
#ifdef INPUT_REQUIRES_GRAD
_input_grad = _output_grad;
#endif
}