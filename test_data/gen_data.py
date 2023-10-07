import stim
import yaml


def surface_code():
    code = "surface_code:rotated_memory_z"
    distance = 21
    rounds = 10
    shots = 1000
    circuit = stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    sampler = circuit.compile_detector_sampler()
    dets = sampler.sample(shots=shots, bit_packed=False)
    stim.write_shot_data_file(
        data=dets,
        path="surface_code/detectors.b8",
        format='b8',
        num_detectors=circuit.num_detectors,
    )
    stim.write_shot_data_file(
        data=dets,
        path="surface_code/detectors.01",
        format='01',
        num_detectors=circuit.num_detectors,
    )
    metadata = {
        "code": code,
        "distance": distance,
        "rounds": rounds,
        "num_shots": shots,
        "num_detectors": circuit.num_detectors,
    }
    with open("surface_code/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


def rep_code():
    code = "repetition_code:memory"
    distance = 5
    rounds = 5
    shots = 1000
    circuit = stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.01,
        after_reset_flip_probability=0.01,
        before_measure_flip_probability=0.01,
        before_round_data_depolarization=0.01,
    )
    sampler = circuit.compile_detector_sampler()
    dets = sampler.sample(shots=shots, bit_packed=False)
    stim.write_shot_data_file(
        data=dets,
        path="rep_code/detectors.b8",
        format='b8',
        num_detectors=circuit.num_detectors,
    )
    stim.write_shot_data_file(
        data=dets,
        path="rep_code/detectors.01",
        format='01',
        num_detectors=circuit.num_detectors,
    )
    metadata = {
        "code": code,
        "distance": distance,
        "rounds": rounds,
        "num_shots": shots,
        "num_detectors": circuit.num_detectors,
    }
    with open("rep_code/metadata.yaml", "w") as f:
        yaml.dump(metadata, f)


if __name__ == '__main__':
    rep_code()
