import stim


def main():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=21,
        rounds=10,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    sampler = circuit.compile_detector_sampler()
    dets = sampler.sample(shots=1000, bit_packed=False)
    stim.write_shot_data_file(
        data=dets,
        path="detectors.b8",
        format='b8',
        num_detectors=circuit.num_detectors,
    )
    stim.write_shot_data_file(
        data=dets,
        path="detectors.01",
        format='01',
        num_detectors=circuit.num_detectors,
    )


if __name__ == '__main__':
    main()
