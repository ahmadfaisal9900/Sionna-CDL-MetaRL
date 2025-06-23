def configure_ris_mode_powers(ris, mode_powers):
    """
    Configure the power distribution across different modes of a multi-mode RIS
    
    Parameters:
    -----------
    ris: RIS
        The reconfigurable intelligent surface object
    mode_powers: tf.Tensor
        A tensor of shape [num_modes] containing the relative power for each mode
        (should sum to 1.0)
    """
    num_modes, num_rows, num_cols = ris.amplitude_profile.values.shape
    
    # Start with uniform amplitude for all elements
    amplitudes = tf.ones([num_modes, num_rows, num_cols])
    
    # Scale each mode by the square root of its allocated power
    # (power is proportional to amplitude squared)
    for mode in range(num_modes):
        amplitudes = tf.tensor_scatter_nd_update(
            amplitudes,
            [[mode, i, j] for i in range(num_rows) for j in range(num_cols)],
            [tf.sqrt(mode_powers[mode])] * (num_rows * num_cols)
        )
    
    # Update the RIS amplitude profile
    ris.amplitude_profile.update(amplitudes)
    
    return ris
