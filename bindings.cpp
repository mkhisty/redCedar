#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include "tensor.hpp"
#include "nn.hpp"

namespace py = pybind11;
using namespace NN;

PYBIND11_MODULE(redCedar, m) {
    // gradNode bindings
    py::class_<gradNode>(m, "gradNode")
        .def(py::init<>())
        .def_readwrite("children", &gradNode::children)
        .def_readwrite("module", &gradNode::module);

    // Tensor bindings
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<int>, bool, std::string>(),
             py::arg("dims"), py::arg("grad") = false, py::arg("preset") = "none")
        .def(py::init([](std::vector<int> dims, py::list data, bool copy, bool grad) {
            std::vector<float> vec_data;
            for (auto item : data) {
                vec_data.push_back(item.cast<float>());
            }
            return new Tensor(dims, vec_data.data(), copy, grad);
        }), py::arg("dims"), py::arg("data"), py::arg("copy") = true, py::arg("grad") = true)
        .def(py::init([](const Tensor& other) {
            return new Tensor(other);
        }), py::arg("other"))
        .def("__getitem__", [](Tensor &self, int index) {
            return self[index];
        })
        .def("__setitem__", [](Tensor &self, int index, float value) {
            self[index] = value;
        })
        .def("__setitem__", [](Tensor &self, int index, const Tensor &other) {
            self[index] = other;
        })
        .def("__setitem__", [](Tensor &self, int index, py::list data) {
            auto sub_dims = self[index].dims();
            if (sub_dims.size() == 1) {
                for (int i = 0; i < data.size() && i < sub_dims[0]; i++) {
                    self[index][i] = data[i].cast<float>();
                }
            } else if (sub_dims.size() == 2) {
                std::vector<std::vector<float>> vec_2d;
                for (auto row : data) {
                    std::vector<float> vec_row;
                    py::list row_list = row.cast<py::list>();
                    for (auto item : row_list) {
                        vec_row.push_back(item.cast<float>());
                    }
                    vec_2d.push_back(vec_row);
                }
                self[index] = vec_2d;
            }
        })
        .def("toFloat", &Tensor::toFloat)
        .def("toList", &Tensor::toList)
        .def("__mul__", [](const Tensor &self, float f) { return self * f; })
        .def("__rmul__", [](const Tensor &self, float f) { return self * f; })
        .def("assign_float", [](Tensor &self, float f) { self = f; })
        .def("assign_vector", [](Tensor &self, const std::vector<std::vector<float>> &v) { self = v; })
        .def("assign", [](Tensor &self, const Tensor &other) { self = other; return self; })
        .def("transpose", &Tensor::transpose)
        .def("matmul", &Tensor::matmul)
        .def("__matmul__", [](const Tensor &a, const Tensor &b) { return a.matmul(b); })
        .def("add", &Tensor::add)
        .def("__add__", [](const Tensor &a, const Tensor &b) { return a.add(b); })
        .def("__xor__", [](const Tensor &a, int power) { return a.operator^(power); })
        .def("dims", &Tensor::dims)
        .def("strides", &Tensor::strides)
        .def_property_readonly("shape", &Tensor::dims)
        .def("sum", &Tensor::sum)
        .def("size", &Tensor::size)
        .def("print", &Tensor::print)
        .def_readwrite("owner", &Tensor::owner)
        .def_readwrite("grad_", &Tensor::grad_)
        .def_readwrite("root_", &Tensor::root_)
        .def("__repr__", [](const Tensor &self) {
            std::string repr = "Tensor(dims=[";
            auto dims = self.dims();
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i > 0) repr += ", ";
                repr += std::to_string(dims[i]);
            }
            repr += "], grad=" + std::string(self.grad_ ? "True" : "False") + ")";
            return repr;
        })
        .def("__str__", [](const Tensor &self) {
            // Capture output from print function
            std::ostringstream oss;
            std::streambuf* orig = std::cout.rdbuf();
            std::cout.rdbuf(oss.rdbuf());

            self.print();

            std::cout.rdbuf(orig);
            return oss.str();
        })
        .def_static("fromList", &Tensor::fromList, py::arg("data"), py::arg("dims"));

    // ----------------
    // NN Bindings
    // ----------------
    py::class_<BaseModule>(m, "BaseModule")
        .def("forward", &BaseModule::forward)
        .def("backward", &BaseModule::backward);

    py::class_<Linear, BaseModule>(m, "Linear")
        .def(py::init<int, int>())
        .def_readwrite("weights", &Linear::weights)
        .def_readwrite("bias", &Linear::bias);

    py::class_<MSELoss, BaseModule>(m, "MSELoss")
        .def(py::init<>())
        .def("forward", [](MSELoss &self, Tensor &input, Tensor &correct) {
            return self.forward(input, correct);
        });

    py::class_<Sigmoid, BaseModule>(m, "Sigmoid")
        .def(py::init<>());

    py::class_<Model>(m, "Model")
        .def(py::init<std::vector<BaseModule*>>())
        .def("forward", &Model::forward)
        .def("backward", &Model::backward);

    py::class_<Optim>(m, "Optim")
        .def(py::init<float>(), py::arg("lr") = 0.01f)
        .def("optimize", [](Optim& self, Tensor& tensor) { self.optimize(tensor); })
        .def_readwrite("lr", &Optim::lr);

}
