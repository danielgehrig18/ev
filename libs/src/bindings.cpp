#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <memory>
#include <iostream>

namespace py = pybind11;

// Example: wrap a std::vector<float> in a py::array_t<float> without copying.
py::array_t<float> wrap_vector_no_copy(std::vector<float>& vec) {
    // Construct a 1D shape and corresponding stride for float data
    py::ssize_t size = static_cast<py::ssize_t>(vec.size());
    py::ssize_t stride = sizeof(float);

    // Create a capsule that does nothing special on destruction
    // (assuming `vec` is owned elsewhere, e.g. static or guaranteed to stay alive)

    // Create pybind11 array:
    //  - shape = (size,)
    //  - strides = (stride,)
    //  - data pointer = vec.data()
    //  - base capsule to tie the lifetime to
    return py::array_t<float>(
        {size},          // shape
        {stride},        // stride
        vec.data()       // data pointer
    );
}


float distance_squared(float* a, float* b, int dimension) {
    float norm_sq = 0;
    for (int i=0; i<dimension; i++) {
        float d = a[i] - b[i];
        norm_sq += d * d;
    }
    return norm_sq;
}

void vector_diff(std::vector<float>& out, float* a, float* b, int dimension) {
    for (int i=0; i<dimension; i++) {
        out.push_back(a[i] - b[i]);
    }
}

float dot_product(std::vector<float>& a, std::vector<float>& b) {
    float out = 0;
    for (int i=0; i<a.size(); i++) {
        out += a[i] * b[i];
    }
    return out;
}

float interpolate_and_update_ref(float* signal_0, float* signal_1, float* reference, int dimension, float th_sq) {
    // solve when |s0 + (s1 - s0) t - r|^2 = th^2
    // call ds = s1 - s0
    // call dr = r - s0 -> |ds
    // then we have |ds t - dr |^2 = th^2
    // so |ds|^2 t^2 - 2 ds^T dr + |dr|^2 - th^2 = 0
    // a = |ds|^2
    // b = ds * dr
    // c = |dr|^2 - th^2 < 0
    // so t = b/a + sqrt((b/a)^2 - c/a)
    // we do not need to check discriminant, because -c/a > 0, since |dr|^2 < th^2
    std::vector<float> dr, ds;
    vector_diff(dr, reference, signal_0, dimension);
    vector_diff(ds, signal_1, signal_0, dimension);
    //std::cout << " blue s0 " << signal_0[0] << " " << signal_0[1] << " s1 " << signal_1[0] << " " << signal_1[1] << " ds " << ds[0] << " " << ds[1] << std::endl;

    float a = dot_product(ds, ds);
    float b = dot_product(ds, dr);
    float c = dot_product(dr, dr) - th_sq;

    float b_a = b/a;
    float c_a = c/a;

    float t = b_a + std::sqrt((b_a)*(b_a) - c_a);

    //std::cout << "s0 " << signal_0[0] << " " << signal_0[1] << " s1 " << signal_1[0] << " " << signal_1[1] << " ds " << ds[0] << " " << ds[1] << std::endl;
    //std::cout << "updating ref from " << reference[0] << " to " << signal_0[0] + t * ds[0] << " with t="<< t << std::endl;
    //std::cout << "updating ref from " << reference[1] << " to " << signal_0[1] + t * ds[1] << " with t="<< t << std::endl;

    for (int i=0; i<dimension; i++) {
        reference[i] = signal_0[i] + t * ds[i];
    }

    return t;
}

/*
 * Example function that:
 *  1) Accepts a NumPy array of shape (n, 2).
 *  2) Returns a new NumPy array of shape (M,).
 */
py::array_t<float> generate_events(const py::array_t<float>& signal, float threshold, py::array_t<float>& reference) {
    //std::cout << "leggo: "<< std::endl;

    // 2. Basic shape checks
    auto buf_signal = signal.request();
    if (buf_signal.ndim != 2) {
        throw std::runtime_error("Signal must have dimension N x D");
    }
    int dimension = buf_signal.shape[1];

    auto buf_ref = reference.request();
    if (buf_ref.ndim != 1) {
        throw std::runtime_error("Reference must have size D.");
    }

    if (buf_ref.shape[0] != dimension) {
        throw std::runtime_error("Reference and signal must have same dimension.");
    }

    size_t n = buf_signal.shape[0];
    float* signal_it = static_cast<float*>(buf_signal.ptr);
    float* ref_it = static_cast<float*>(buf_ref.ptr);

    std::vector<float> output;
    float threshold_sq = threshold * threshold;

    for (int i=0; i<n; i++) {
        while (distance_squared(signal_it, ref_it, dimension) > threshold_sq) {
            float j = interpolate_and_update_ref(signal_it - dimension, signal_it, ref_it, dimension, threshold_sq);
            output.push_back(i - 1 + j);
            //std::cout << "i=" << i << " logged: " << i-1+j << " distance: " << distance_squared(signal_it, ref_it, dimension) << std::endl;
        }
        signal_it += dimension;
    }
    return wrap_vector_no_copy(output);
}


// Create the module
PYBIND11_MODULE(ev_cpp, m) {
    m.doc() = "Pybind11 extension to generate n-dimensional events.";
    m.def("generate_events", &generate_events, "Generate events");
}
