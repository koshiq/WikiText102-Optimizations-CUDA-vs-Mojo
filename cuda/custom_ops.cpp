#include <torch/extension.h>
#include <vector>

// Forward declarations for CUDA host functions
torch::Tensor gemm_cuda_fp32(torch::Tensor A, torch::Tensor B, float alpha, float beta);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_cuda_forward(
    torch::Tensor input, torch::Tensor gamma, torch::Tensor beta, float eps);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layer_norm_cuda_backward(
    torch::Tensor grad_output, torch::Tensor input, torch::Tensor gamma,
    torch::Tensor mean, torch::Tensor rstd);

torch::Tensor softmax_cuda_forward(torch::Tensor input, int dim);

torch::Tensor softmax_cuda_backward(
    torch::Tensor grad_output, torch::Tensor output);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor gemm_forward(torch::Tensor input, torch::Tensor weights) {
    CHECK_INPUT(input);
    CHECK_INPUT(weights);
    return gemm_cuda_fp32(input, weights.t().contiguous(), 1.0, 0.0);
}

class CustomSoftmaxAutograd : public torch::autograd::Function<CustomSoftmaxAutograd> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, int64_t dim) {
        CHECK_INPUT(input);
        auto output = softmax_cuda_forward(input, dim);
        ctx->save_for_backward({output});
        return output;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto output = saved[0];
        auto grad_output = grad_outputs[0];
        auto grad_input = softmax_cuda_backward(grad_output, output);
        return {grad_input, torch::autograd::Variable()};
    }
};

torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    return CustomSoftmaxAutograd::apply(input, dim);
}

class CustomLayerNormAutograd : public torch::autograd::Function<CustomLayerNormAutograd> {
public:
    static torch::Tensor forward(torch::autograd::AutogradContext *ctx, torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps) {
        CHECK_INPUT(input);
        CHECK_INPUT(weight);
        CHECK_INPUT(bias);

        auto [output, mean, rstd] = layer_norm_cuda_forward(input, weight, bias, eps);
        ctx->save_for_backward({input, weight, mean, rstd});
        return output;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext *ctx, torch::autograd::variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto mean = saved[2];
        auto rstd = saved[3];
        auto grad_output = grad_outputs[0];

        auto [grad_input, grad_weight, grad_bias] = layer_norm_cuda_backward(grad_output, input, weight, mean, rstd);

        return {grad_input, grad_weight, grad_bias, torch::autograd::Variable()};
    }
};

torch::Tensor layernorm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps) {
    return CustomLayerNormAutograd::apply(input, weight, bias, eps);
}

class CustomLinear : public torch::nn::Module {
public:
    CustomLinear(int64_t in_features, int64_t out_features) {
        weight = register_parameter("weight", torch::randn({out_features, in_features}));
        bias = register_parameter("bias", torch::randn({out_features}));
    }

    torch::Tensor forward(torch::Tensor input) {
        // Support batched sequence inputs with shape [seq_len, batch, features]
        if (input.dim() == 3) {
            auto sizes = input.sizes(); // [seq_len, batch, features]
            int64_t seq_len = sizes[0];
            int64_t batch = sizes[1];
            int64_t features = sizes[2];
            auto inp = input.contiguous().view({seq_len * batch, features});
            auto out = gemm_forward(inp, weight); // [seq_len*batch, out_features]
            int64_t out_features = out.size(1);
            out = out.view({seq_len, batch, out_features});
            // bias has shape [out_features], make it broadcastable to [seq_len, batch, out_features]
            return out + bias.view({1, 1, out_features});
        }

        // Fallback for 2D inputs [batch, features]
        auto out = gemm_forward(input, weight);
        return out + bias;
    }

    int64_t get_in_features() const {
        return weight.size(1);
    }

    int64_t get_out_features() const {
        return weight.size(0);
    }

    torch::Tensor get_weight() {
        return weight;
    }

    torch::Tensor get_bias() {
        return bias;
    }

    void set_weight(torch::Tensor w) {
        weight = w;
    }

    void set_bias(torch::Tensor b) {
        bias = b;
    }

    void cuda_() {
        weight = weight.cuda();
        bias = bias.cuda();
    }

    void cpu_() {
        weight = weight.cpu();
        bias = bias.cpu();
    }

private:
    torch::Tensor weight, bias;
};

class CustomSoftmax : public torch::nn::Module {
public:
    CustomSoftmax(int64_t dim = -1) : dim(dim) {}

    torch::Tensor forward(torch::Tensor input) {
        return softmax_forward(input, dim);
    }

private:
    int64_t dim;
};

class CustomLayerNorm : public torch::nn::Module {
public:
    CustomLayerNorm(torch::IntArrayRef normalized_shape, double eps = 1e-5) : eps(eps) {
        // In this simplified model, we assume normalized_shape is a single integer.
        // A full implementation would handle arbitrary shapes.
        if (normalized_shape.size() != 1) {
            throw std::runtime_error("CustomLayerNorm only supports 1D normalized_shape");
        }
        weight = register_parameter("weight", torch::ones(normalized_shape));
        bias = register_parameter("bias", torch::zeros(normalized_shape));
    }

    torch::Tensor forward(torch::Tensor input) {
        return layernorm_forward(input, weight, bias, eps);
    }

private:
    torch::Tensor weight, bias;
    double eps;
public:
    void cuda_() {
        weight = weight.cuda();
        bias = bias.cuda();
    }

    void cpu_() {
        weight = weight.cpu();
        bias = bias.cpu();
    }

    torch::Tensor get_weight() {
        return weight;
    }

    torch::Tensor get_bias() {
        return bias;
    }
};

bool use_custom_ops() {
    return true;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_forward", &gemm_forward, "Custom GEMM forward (CUDA)");
    m.def("softmax_forward", &softmax_forward, "Custom Softmax forward (CUDA)", py::arg("input"), py::arg("dim") = -1);
    m.def("layernorm_forward", &layernorm_forward, "Custom LayerNorm forward (CUDA)", py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
    m.def("use_custom_ops", &use_custom_ops, "Check if custom ops are available");

    py::class_<CustomLinear, std::shared_ptr<CustomLinear>>(m, "CustomLinear")
        .def(py::init<int64_t, int64_t>())
        .def("forward", &CustomLinear::forward)
        .def("__call__", &CustomLinear::forward)
        .def("cuda", &CustomLinear::cuda_)
        .def("cpu", &CustomLinear::cpu_)
        .def_property_readonly("in_features", &CustomLinear::get_in_features)
        .def_property_readonly("out_features", &CustomLinear::get_out_features)
        .def_property("weight", &CustomLinear::get_weight, &CustomLinear::set_weight)
        .def_property("bias", &CustomLinear::get_bias, &CustomLinear::set_bias);

    py::class_<CustomSoftmax, std::shared_ptr<CustomSoftmax>>(m, "CustomSoftmax")
        .def(py::init<int64_t>(), py::arg("dim") = -1)
        .def("forward", &CustomSoftmax::forward)
        .def("__call__", &CustomSoftmax::forward);
        

    py::class_<CustomLayerNorm, std::shared_ptr<CustomLayerNorm>>(m, "CustomLayerNorm")
        .def(py::init([](int64_t normalized_shape, double eps) {
            return new CustomLayerNorm(torch::IntArrayRef(std::vector<int64_t>{normalized_shape}), eps);
        }), py::arg("normalized_shape"), py::arg("eps") = 1e-5)
        .def("forward", &CustomLayerNorm::forward)
        .def("__call__", &CustomLayerNorm::forward)
        .def("cuda", &CustomLayerNorm::cuda_)
        .def("cpu", &CustomLayerNorm::cpu_)
        .def_property_readonly("weight", &CustomLayerNorm::get_weight)
        .def_property_readonly("bias", &CustomLayerNorm::get_bias);
}