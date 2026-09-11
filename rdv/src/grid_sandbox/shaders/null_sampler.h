FORWARD {
    _output[0] = _input[0] * 0.5 + _input[1] * 0.3 + _input[2] * 0.2;
}

BACKWARD {
    NOT_SUPPORTED("null_sampler backward not supported");
}
