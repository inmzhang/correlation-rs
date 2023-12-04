import pathlib

import stim
import pymatching
import numpy as np

import correlation_py as cpy


def main():
    data_dir = pathlib.Path("/Users/inm/paper_data/google_qec_d3_5/google_qec3v5_experiment_data/surface_code_bX_d3_r03_center_7_5")
    google_derived_dem_filepath = data_dir / "pij_from_even_for_odd.dem"
    google_derived_dem = stim.DetectorErrorModel.from_file(google_derived_dem_filepath)

    tanner_graph = cpy.TannerGraph(google_derived_dem)
    google_bootstrap_dem = tanner_graph.to_detetor_error_model()
    # sampler = google_derived_dem.compile_sampler()
    # detectors, _, _ = sampler.sample(5000000)
    detectors = stim.read_shot_data_file(path=data_dir/"detection_events.b8", format="b8", num_detectors=24)
    detectors = detectors[::2]

    # correlation_results = cpy.cal_2nd_order_correlations(detectors)
    correlation_results = cpy.cal_high_order_correlations(detectors, tanner_graph.hyperedges)
    nonnegative_probs = {
        h: p if p > 0 else tanner_graph.hyperedge_probs.get(h)
        for h, p in correlation_results.data.items()
    }
    correlation_dem = tanner_graph.with_probs(nonnegative_probs).to_detetor_error_model()


    obs_flips = stim.read_shot_data_file(path=data_dir/"obs_flips_actual.01", format="01", num_observables=1)
    # decode using Google's derived DEM
    matching = pymatching.Matching.from_detector_error_model(google_derived_dem)
    predicted_obs_flips = matching.decode_batch(detectors)
    print(f"Google derived DEM: {np.mean(obs_flips[::2] == predicted_obs_flips)}")

    # decode using Google's circuit dem
    matching = pymatching.Matching.from_detector_error_model_file(str(data_dir/"circuit_detector_error_model.dem"))
    predicted_obs_flips = matching.decode_batch(detectors)
    print(f"Google circuit DEM: {np.mean(obs_flips[::2] == predicted_obs_flips)}")
    
    # bootstrap
    matching = pymatching.Matching.from_detector_error_model(google_bootstrap_dem)
    predicted_obs_flips = matching.decode_batch(detectors)
    print(f"Google bootstrap DEM: {np.mean(obs_flips[::2] == predicted_obs_flips)}")

    # mine
    matching = pymatching.Matching.from_detector_error_model(correlation_dem)
    predicted_obs_flips = matching.decode_batch(detectors)
    print(f"My DEM: {np.mean(obs_flips[::2] == predicted_obs_flips)}")
    


    # aim = frozenset([17, 5, 7])
    # for hyperedge in hyperedges:
    #     if len(hyperedge) == 3:
    #         print(f"Hyperedge: {hyperedge}, ideal: {tanner_graph.hyperedge_probs.get(hyperedge)}, result: {correlation_results.get(hyperedge)}")
        
    

if __name__ == '__main__':
    main()
    

