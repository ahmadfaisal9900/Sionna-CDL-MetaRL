def create_trainable_ris(scene, name, position, orientation, size):
    """Create a RIS with trainable phase and amplitude profiles"""
    wavelength = 3e8/scene.frequency
    num_rows = int(size[0]/(0.5*wavelength))
    num_cols = int(size[1]/(0.5*wavelength))
    
    # Create RIS
    ris = RIS(name=name,
              position=position,
              orientation=orientation,
              num_rows=num_rows,
              num_cols=num_cols)
    
    # Initialize phase profile as trainable variable
    phases = tf.Variable(
        tf.zeros([1, num_rows, num_cols]),
        trainable=True,
        name=f"{name}_phases"
    )
    
    # Make amplitudes trainable if desired
    amplitudes = tf.Variable(
        tf.ones([1, num_rows, num_cols]),
        trainable=True, 
        name=f"{name}_amplitudes"
    )
    
    # Update the RIS profiles
    ris.phase_profile.update(phases)
    ris.amplitude_profile.update(amplitudes)
    
    # Add to scene
    scene.add(ris)
    
    return ris
