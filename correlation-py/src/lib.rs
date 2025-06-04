use correlation::{cal_2nd_order_correlation, cal_high_order_correlations};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyFrozenSet};

#[pyfunction]
fn cal_2nd_order_correlation_rs<'py>(
    py: Python<'py>,
    detection_events: PyReadonlyArray2<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    let detection_events = detection_events.as_array().to_owned();
    let (boundaries, edges) = cal_2nd_order_correlation(&detection_events);
    Ok((boundaries.into_pyarray(py), edges.into_pyarray(py)))
}

#[pyfunction]
fn cal_high_order_correlations_rs<'py>(
    py: Python<'py>,
    detection_events: PyReadonlyArray2<'py, f64>,
    hyperedges: Option<Vec<Vec<usize>>>,
    num_threads: Option<usize>,
    max_iters: Option<u64>,
) -> PyResult<Vec<(Py<PyFrozenSet>, f64)>> {
    let detection_events = detection_events.as_array().to_owned();
    let results = cal_high_order_correlations(
        &detection_events,
        hyperedges.as_deref(),
        num_threads,
        max_iters,
    )
    .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?
    .into_iter()
    .map(|(k, v)| {
        (
            PyFrozenSet::new(py, k.iter().collect::<Vec<_>>())
                .unwrap()
                .into(),
            v,
        )
    })
    .collect();
    Ok(results)
}

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(cal_2nd_order_correlation_rs))?;
    m.add_wrapped(wrap_pyfunction!(cal_high_order_correlations_rs))?;
    Ok(())
}
