import pathlib

import stim

from correlation_py import cal_2nd_order_correlations, cal_high_order_correlations


def main():
    save_dir = pathlib.Path(
        "/Users/inm/Programming/RustProject/correlation-rs/correlation/test_data/rep_code"
    )
    dets = stim.read_shot_data_file(
        path=save_dir / "detectors.b8",
        format="b8",
        num_detectors=24,
    )

    res = cal_2nd_order_correlations(dets)
    bdy, edges = res.data
    print(bdy)
    print(edges)

    result = cal_high_order_correlations(dets)
    print(result)


if __name__ == "__main__":
    main()
