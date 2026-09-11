/* Diagnostic: the cheapest possible sampler. Reads the three input coordinates,
does one multiply-add, writes one output. No index math, no memory fetch. Timing
this gives the floor set by dispatching the threads and streaming the input and
output tensors, with essentially no sampling work on top. */

FORWARD {
    _output[0] = _input[0] * 0.5 + _input[1] * 0.3 + _input[2] * 0.2;
}

BACKWARD {
    NOT_SUPPORTED("null_sampler backward not supported");
}
